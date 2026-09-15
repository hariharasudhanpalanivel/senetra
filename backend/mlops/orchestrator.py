"""Continuous training and canary control loop for one backend deployment."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

from senetra_ml.config import PipelineConfig
from senetra_ml.data.repository import SenetraRepository
from senetra_ml.evaluation.online import compare_on_fresh_data, fresh_window_panel
from senetra_ml.models.bundle import ForecastBundle
from senetra_ml.tracking import RegistryModelLoader

from mlops.canary import VariantStats, decide
from mlops.data_watch import DataSignature, decide_retrain, read_signature
from mlops.deployment import DeploymentManager
from mlops.jobs import JobRunner
from mlops.settings import MLOpsSettings
from mlops.state import StateStore

log = logging.getLogger(__name__)


class MLOrchestrator:
    def __init__(self, cfg: PipelineConfig, settings: MLOpsSettings, state: StateStore,
                 deployments: DeploymentManager, runner: JobRunner):
        self.cfg = cfg
        self.settings = settings
        self.state = state
        self.deployments = deployments
        self.runner = runner
        self._loader = RegistryModelLoader(cfg)
        self._bundles: dict[tuple[str, str], ForecastBundle] = {}
        now = time.monotonic()
        self._next_due = {"data": now, "canary": now, "monitor": now + settings.monitor_interval_seconds}
        self._lock = threading.RLock()

    def targets(self) -> list[str]:
        return list(self.settings.retrain_targets or self.cfg.targets)

    # ------------------------------------------------------------------ new data -> retraining

    def _decision(self, current: DataSignature, trained: DataSignature | None, force: bool):
        last_job = self.state.last_finished_job("retrain")
        finished = datetime.fromisoformat(last_job["finished_at"]) if last_job else None
        canary_active = any(d.canary_version for d in self.deployments.all_status(self.targets()))
        return decide_retrain(current, trained, self.settings, last_finished_at=finished,
                              canary_active=canary_active, force=force, now=datetime.now(timezone.utc))

    def data_status(self) -> dict:
        current = read_signature(self.cfg.data.db_path)
        last = self.state.last_signature()
        trained = DataSignature.from_json(last["signature"]) if last else None
        if trained is None and any(d.stable_version for d in self.deployments.all_status(self.targets())):
            would_retrain, reason = False, ("champion models exist but no baseline is recorded; the next data check "
                                            "records the current data as the baseline without retraining")
        else:
            decision = self._decision(current, trained, force=False)
            would_retrain, reason = decision.should_retrain, decision.reason
        return {
            "current": current.as_dict(),
            "last_trained": ({**trained.as_dict(), "recorded_at": last["recorded_at"], "job_id": last["job_id"],
                              "baseline": bool(last["baseline"])} if trained else None),
            "would_retrain": would_retrain,
            "reason": reason,
        }

    def check_for_new_data(self, trigger: str = "schedule", force: bool = False,
                           targets: list[str] | None = None) -> dict:
        with self._lock:
            current = read_signature(self.cfg.data.db_path)
            last = self.state.last_signature()
            if last is None and not force and any(d.stable_version for d in self.deployments.all_status(self.targets())):
                self.state.record_signature(current.to_json(), current.latest_date, job_id=None, baseline=True)
                return {"action": "baseline_recorded",
                        "reason": "champion models already exist; current data recorded as the training baseline"}
            trained = DataSignature.from_json(last["signature"]) if last else None
            decision = self._decision(current, trained, force)
            if not decision.should_retrain:
                return {"action": "none", "reason": decision.reason}
            job = self.runner.submit_retrain(trigger, targets or self.targets(), force, current)
            return {"action": "retrain", "reason": decision.reason, "new_days": decision.new_days, "job": job}

    # ------------------------------------------------------------------ canary evaluation

    def _bundle(self, target: str, version: str) -> ForecastBundle:
        key = (target, str(version))
        if key not in self._bundles:
            self._bundles[key] = self._loader.load_version(target, version).bundle
        return self._bundles[key]

    def accuracy(self, target: str, stable_version: str | None, canary_version: str) -> dict:
        if not stable_version:
            return {"status": "pending", "reason": "no stable model to compare with"}
        stable, canary = self._bundle(target, stable_version), self._bundle(target, canary_version)
        after = max(pd.Timestamp(b.metadata["training_end"]) for b in (stable, canary))
        with SenetraRepository(self.cfg.data.db_path) as repo:
            panel = fresh_window_panel(self.cfg, repo, after)
        result = compare_on_fresh_data(self.cfg, panel, target, {"stable": stable, "canary": canary}, after)
        if result["status"] != "compared":
            return result
        models = result["models"]
        return {
            "status": "compared", "after_date": result["after_date"], "rows": models["canary"]["rows"],
            "stable_district_wape": models["stable"]["district_wape"],
            "canary_district_wape": models["canary"]["district_wape"],
            "stable_phc_mae": models["stable"]["phc_mae"], "canary_phc_mae": models["canary"]["phc_mae"],
        }

    def evaluate_canary(self, target: str) -> dict:
        deployment = self.deployments.status(target, refresh=True)
        if not deployment.canary_version:
            return {"target": target, "action": "none", "reason": "no active canary"}
        since = deployment.canary_stage_started_at
        canary = VariantStats.from_dict(self.state.prediction_stats(target, "canary", deployment.canary_version, since))
        stable = (VariantStats.from_dict(self.state.prediction_stats(target, "stable", deployment.stable_version, since))
                  if deployment.stable_version else VariantStats())
        accuracy = self.accuracy(target, deployment.stable_version, deployment.canary_version)
        decision = decide(deployment, canary, stable, accuracy, self.settings, datetime.now(timezone.utc))
        self.state.add_event(target, deployment.canary_version, f"evaluate:{decision.action}",
                             deployment.canary_percent, {"reason": decision.reason, "checks": decision.checks})
        outcome = None
        if decision.action == "advance":
            outcome = self.deployments.advance(target, decision.reason)
        elif decision.action == "promote":
            outcome = self.deployments.promote(target, decision.reason)
        elif decision.action == "rollback":
            outcome = self.deployments.rollback(target, decision.reason)
        log.info("canary %s v%s: %s (%s)", target, deployment.canary_version, decision.action, decision.reason)
        return {"target": target, "version": deployment.canary_version, "action": decision.action,
                "reason": decision.reason, "checks": decision.checks, "outcome": outcome}

    def evaluate_canaries(self) -> list[dict]:
        return [self.evaluate_canary(d.target) for d in self.deployments.all_status(self.targets(), refresh=True)
                if d.canary_version]

    # ------------------------------------------------------------------ monitoring

    def run_monitoring(self) -> dict:
        from senetra_ml.pipelines.monitor import run_monitoring

        champions = [d.target for d in self.deployments.all_status(self.targets()) if d.stable_version]
        if not champions:
            return {"status": "skipped", "reason": "no champion models yet"}
        report = run_monitoring(self.cfg, champions)
        return {"status": report["status"], "generated_at": report["generated_at"],
                "alerts": sum(a["severity"] == "alert" for a in report["alerts"]),
                "warnings": sum(a["severity"] == "warn" for a in report["alerts"])}

    # ------------------------------------------------------------------ scheduling

    def tick(self) -> list[str]:
        self.runner.reap()
        performed = []
        tasks = (
            ("data", self.settings.data_check_interval_seconds, lambda: self.check_for_new_data("schedule")),
            ("canary", self.settings.canary_check_interval_seconds, self.evaluate_canaries),
            ("monitor", self.settings.monitor_interval_seconds, self.run_monitoring),
        )
        for name, interval, task in tasks:
            now = time.monotonic()
            if now < self._next_due[name]:
                continue
            self._next_due[name] = now + interval
            try:
                task()
                performed.append(name)
            except Exception:  # a failing task must not stop the loop
                log.exception("scheduled ML task %r failed", name)
        return performed


class MLScheduler(threading.Thread):
    """Runs `MLOrchestrator.tick` in the one backend process that holds the scheduler lease."""

    LEASE = "senetra-ml-scheduler"

    def __init__(self, orchestrator: MLOrchestrator, state: StateStore, settings: MLOpsSettings):
        super().__init__(name="senetra-ml-scheduler", daemon=True)
        self.orchestrator = orchestrator
        self.state = state
        self.settings = settings
        self.owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._stop_event = threading.Event()
        self.is_leader = False

    def run(self) -> None:
        ttl = max(60.0, self.settings.scheduler_poll_seconds * 4)
        while not self._stop_event.is_set():
            try:
                self.is_leader = self.state.try_acquire_lease(self.LEASE, self.owner, ttl)
                if self.is_leader:
                    self.orchestrator.tick()
            except Exception:
                log.exception("ML scheduler iteration failed")
            self._stop_event.wait(self.settings.scheduler_poll_seconds)
        self.state.release_lease(self.LEASE, self.owner)

    def stop(self) -> None:
        self._stop_event.set()
