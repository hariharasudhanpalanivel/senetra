"""Retraining jobs.

With `job_runner = "process"` each job runs in its own OS process (multiprocessing spawn), so the
training memory never lives inside a web worker and a crash cannot take the API down. The job writes
its progress to the shared state store; the runner only starts processes and reaps them.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import threading
import traceback
from datetime import datetime, timedelta, timezone

from senetra_ml.config import PipelineConfig

from mlops.data_watch import DataSignature
from mlops.settings import MLOpsSettings
from mlops.state import StateStore, utcnow

log = logging.getLogger(__name__)
HEARTBEAT_SECONDS = 30


def _heartbeat(state: StateStore, job_id: int, stop: threading.Event) -> None:
    while not stop.wait(HEARTBEAT_SECONDS):
        state.update_job(job_id, heartbeat_at=utcnow())


def run_retrain_job(job_id: int, cfg: PipelineConfig, settings: MLOpsSettings, targets: list[str] | None,
                    signature_json: str, trigger: str) -> None:
    """validate -> train -> evaluate (gates, no auto-promotion) -> canary for passing models."""
    from senetra_ml.data.repository import SenetraRepository
    from senetra_ml.data.validation import validate_database
    from senetra_ml.pipelines.evaluate import run_evaluation
    from senetra_ml.pipelines.train import run_training

    from mlops.deployment import DeploymentManager

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    state = StateStore(settings.state_db)
    state.update_job(job_id, status="running", started_at=utcnow(), heartbeat_at=utcnow(), pid=os.getpid())
    stop = threading.Event()
    threading.Thread(target=_heartbeat, args=(state, job_id, stop), daemon=True).start()
    try:
        with SenetraRepository(cfg.data.db_path) as repo:
            report = validate_database(repo, cfg)
        if report.errors:
            raise RuntimeError("data validation failed: " + "; ".join(f"{c.name}: {c.detail}" for c in report.errors))

        log.info("retrain job %s: training %s", job_id, targets or "all targets")
        training = run_training(cfg, targets)
        evaluation = run_evaluation(cfg, training["run_id"], targets=targets, promote=False)

        deployments = DeploymentManager(cfg, settings, state)
        outcomes = {}
        for result in evaluation["results"]:
            target, version = result["target"], str(result["model_version"])
            reason = f"retrain job {job_id} ({trigger})"
            if result["gates_passed"]:
                outcomes[target] = deployments.start_canary(target, version, reason)
            else:
                failed = [g["name"] for g in result["gates"] if g["passed"] is False]
                outcomes[target] = deployments.reject(target, version, f"{reason}: failed gates {failed}")

        signature = DataSignature.from_json(signature_json)
        state.record_signature(signature_json, signature.latest_date, job_id)
        state.update_job(job_id, status="succeeded", finished_at=utcnow(), result=json.dumps({
            "train_run_id": training["run_id"],
            "evaluation_run_id": evaluation["run_id"],
            "trained_through": signature.latest_date,
            "deployments": outcomes,
            "district_wape": {r["target"]: r["district_wape_observed"] for r in evaluation["results"]},
        }, default=str))
        log.info("retrain job %s succeeded: %s", job_id, {t: o["action"] for t, o in outcomes.items()})
    except Exception as exc:  # the job must always end in a terminal state
        log.exception("retrain job %s failed", job_id)
        state.update_job(job_id, status="failed", finished_at=utcnow(),
                         error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    finally:
        stop.set()


class JobRunner:
    def __init__(self, cfg: PipelineConfig, settings: MLOpsSettings, state: StateStore):
        self.cfg = cfg
        self.settings = settings
        self.state = state
        self._processes: dict[int, mp.process.BaseProcess] = {}
        self._lock = threading.RLock()

    def submit_retrain(self, trigger: str, targets: list[str] | None, force: bool, signature: DataSignature) -> dict:
        with self._lock:
            self.reap()
            active = self.state.active_job("retrain")
            if active:
                return {**active, "note": "a retraining job is already running"}
            job_id = self.state.create_job("retrain", trigger, targets, force, signature.to_json())
            args = (job_id, self.cfg, self.settings, list(targets) if targets else None, signature.to_json(), trigger)
            if self.settings.job_runner == "inline":
                run_retrain_job(*args)
            else:
                process = mp.get_context("spawn").Process(target=run_retrain_job, args=args,
                                                          name=f"senetra-retrain-{job_id}")
                process.start()
                self.state.update_job(job_id, pid=process.pid)
                self._processes[job_id] = process
                log.info("retrain job %s started in process %s", job_id, process.pid)
            return self.state.get_job(job_id)

    def reap(self) -> None:
        """Close finished processes and fail jobs whose worker died without reporting."""
        with self._lock:
            for job_id, process in list(self._processes.items()):
                if process.is_alive():
                    continue
                process.join(timeout=0)
                self._processes.pop(job_id)
                job = self.state.get_job(job_id)
                if job and job["status"] in ("queued", "running"):
                    self.state.update_job(job_id, status="failed", finished_at=utcnow(),
                                          error=f"worker process exited with code {process.exitcode} before reporting")
            active = self.state.active_job("retrain")
            if active and active["id"] not in self._processes:
                last_beat = datetime.fromisoformat(active["heartbeat_at"] or active["created_at"])
                if datetime.now(timezone.utc) - last_beat > timedelta(minutes=self.settings.job_stale_minutes):
                    self.state.update_job(active["id"], status="failed", finished_at=utcnow(),
                                          error=f"no heartbeat for {self.settings.job_stale_minutes} minutes")

    def running_processes(self) -> list[int]:
        with self._lock:
            return [job_id for job_id, process in self._processes.items() if process.is_alive()]
