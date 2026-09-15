"""Wires MLOps into a Flask app: shared components in `app.extensions`, scheduler start-up."""

from __future__ import annotations

import atexit
import logging
import multiprocessing
import threading
from dataclasses import dataclass, field

from flask import Flask, current_app

from senetra_ml.config import PipelineConfig, load_config

from mlops.deployment import DeploymentManager
from mlops.jobs import JobRunner
from mlops.orchestrator import MLOrchestrator, MLScheduler
from mlops.predictor import CanaryPredictor
from mlops.settings import MLOpsSettings
from mlops.state import StateStore

log = logging.getLogger(__name__)
EXTENSION_KEY = "senetra_mlops"


@dataclass
class MLOps:
    cfg: PipelineConfig
    settings: MLOpsSettings
    state: StateStore
    deployments: DeploymentManager
    runner: JobRunner
    orchestrator: MLOrchestrator
    scheduler: MLScheduler | None = None
    _predictor: CanaryPredictor | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def predictor(self) -> CanaryPredictor:
        with self._lock:
            if self._predictor is None:
                self._predictor = CanaryPredictor(self.cfg, self.state, self.deployments)
            return self._predictor

    def scheduler_info(self) -> dict:
        return {
            "enabled": self.settings.scheduler_enabled,
            "running_here": bool(self.scheduler and self.scheduler.is_alive()),
            "leader_here": bool(self.scheduler and self.scheduler.is_leader),
            "lease": self.state.lease_owner(MLScheduler.LEASE),
            "job_processes_here": self.runner.running_processes(),
        }


def init_mlops(app: Flask) -> MLOps:
    settings = app.config.get("MLOPS_SETTINGS") or MLOpsSettings.from_env()
    cfg = app.config.get("MLOPS_PIPELINE_CONFIG") or load_config()
    state = StateStore(settings.state_db)
    deployments = DeploymentManager(cfg, settings, state)
    runner = JobRunner(cfg, settings, state)
    orchestrator = MLOrchestrator(cfg, settings, state, deployments, runner)
    mlops = MLOps(cfg, settings, state, deployments, runner, orchestrator)
    app.extensions[EXTENSION_KEY] = mlops

    # Never start the loop in tests or inside a spawned retraining process that re-imported the app.
    if settings.scheduler_enabled and not app.config.get("TESTING") and multiprocessing.parent_process() is None:
        mlops.scheduler = MLScheduler(orchestrator, state, settings)
        mlops.scheduler.start()
        atexit.register(mlops.scheduler.stop)
        log.info("ML scheduler started (%s)", mlops.scheduler.owner)
    return mlops


def get_mlops() -> MLOps:
    return current_app.extensions[EXTENSION_KEY]
