"""MLflow tracking, data lineage and model-registry helpers."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from senetra_ml import __version__
from senetra_ml.config import PipelineConfig
from senetra_ml.data.db import database_fingerprint
from senetra_ml.data.repository import SenetraRepository
from senetra_ml.models.bundle import ForecastBundle

log = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """No registered model version is available for a target."""


def setup_mlflow(cfg: PipelineConfig) -> MlflowClient:
    uri = cfg.tracking_uri()
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


def experiment_id(cfg: PipelineConfig, stage: str) -> str:
    name = cfg.experiment_name(stage)
    existing = mlflow.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    root = cfg.artifact_root()
    return mlflow.create_experiment(name, artifact_location=f"{root}/{name}" if root else None)


def _git_commit(cfg: PipelineConfig) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cfg.home, capture_output=True,
                                text=True, timeout=5, check=False)
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def lineage_tags(cfg: PipelineConfig, repo: SenetraRepository) -> dict[str, str]:
    """Identifies the exact data and code a run used, without copying any data."""
    lo, hi = repo.date_bounds()
    return {
        "data.source": f"sqlite:{cfg.data.db_path.name}",
        "data.fingerprint": database_fingerprint(cfg.data.db_path),
        "data.start": lo.strftime("%Y-%m-%d"),
        "data.end": hi.strftime("%Y-%m-%d"),
        "code.version": __version__,
        "code.git_commit": _git_commit(cfg),
    }


@dataclass
class LoadedModel:
    target: str
    bundle: ForecastBundle
    name: str
    version: str
    alias: str | None
    run_id: str | None

    @property
    def label(self) -> str:
        return f"{self.name}:v{self.version}"


class RegistryModelLoader:
    def __init__(self, cfg: PipelineConfig, alias: str | None = None, allow_latest: bool = False):
        self.cfg = cfg
        self.alias = alias or cfg.mlflow.champion_alias
        self.allow_latest = allow_latest

    def load(self, target: str) -> LoadedModel:
        client = setup_mlflow(self.cfg)
        name = self.cfg.registered_model_name(target)
        try:
            version = client.get_model_version_by_alias(name, self.alias)
            uri, alias = f"models:/{name}@{self.alias}", self.alias
        except MlflowException:
            if not self.allow_latest:
                raise ModelUnavailableError(
                    f"{name} has no '{self.alias}' version; run `senetra-ml train` then `senetra-ml evaluate`"
                ) from None
            try:
                versions = client.search_model_versions(f"name = '{name}'")
            except MlflowException:
                versions = []
            if not versions:
                raise ModelUnavailableError(f"{name} has no registered versions; run `senetra-ml train`") from None
            version = max(versions, key=lambda v: int(v.version))
            uri, alias = f"models:/{name}/{version.version}", None
        model = mlflow.pyfunc.load_model(uri)
        bundle = model.unwrap_python_model().bundle
        log.info("Loaded %s version %s (%s)", name, version.version, alias or "latest")
        return LoadedModel(target, bundle, name, str(version.version), alias, version.run_id)

    def load_version(self, target: str, version: str | int) -> LoadedModel:
        """Load an exact registered version (used to compare champion and canary side by side)."""
        client = setup_mlflow(self.cfg)
        name = self.cfg.registered_model_name(target)
        try:
            model_version = client.get_model_version(name, str(version))
        except MlflowException:
            raise ModelUnavailableError(f"{name} version {version} does not exist") from None
        model = mlflow.pyfunc.load_model(f"models:/{name}/{model_version.version}")
        return LoadedModel(target, model.unwrap_python_model().bundle, name, str(model_version.version),
                           None, model_version.run_id)
