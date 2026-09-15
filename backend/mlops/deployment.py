"""Model deployments backed by the MLflow registry.

Aliases per registered model `senetra-forecast-<target>`:
- `champion`          stable production model
- `canary`            candidate receiving a share of traffic
- `previous_champion` last champion, for instant revert
- `challenger`        candidate that failed the evaluation gates
The canary stage lives in tags on the canary version (`canary.*`), so every backend instance sees the
same rollout state and it is visible in the MLflow UI.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import asdict, dataclass

from mlflow.exceptions import MlflowException

from senetra_ml.config import PipelineConfig
from senetra_ml.tracking import setup_mlflow

from mlops.settings import MLOpsSettings
from mlops.state import StateStore, utcnow

CANARY_ALIAS = "canary"
PREVIOUS_ALIAS = "previous_champion"
VARIANTS = ("stable", "canary")


def traffic_bucket(target: str, version: str, routing_key: str) -> int:
    """Stable 0-99 bucket: the same caller keeps seeing the same model during a rollout."""
    digest = hashlib.sha256(f"{target}|{version}|{routing_key}".encode()).hexdigest()
    return int(digest[:8], 16) % 100


@dataclass
class TargetDeployment:
    target: str
    model_name: str
    stable_version: str | None = None
    canary_version: str | None = None
    previous_version: str | None = None
    canary_percent: int = 0
    canary_stage_index: int = -1
    canary_started_at: str | None = None
    canary_stage_started_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class DeploymentManager:
    CACHE_SECONDS = 10.0

    def __init__(self, cfg: PipelineConfig, settings: MLOpsSettings, state: StateStore):
        self.cfg = cfg
        self.settings = settings
        self.state = state
        self._cache: dict[str, tuple[float, TargetDeployment]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ reading

    def _client(self):
        return setup_mlflow(self.cfg)

    @staticmethod
    def _alias(client, name: str, alias: str) -> str | None:
        try:
            return str(client.get_model_version_by_alias(name, alias).version)
        except MlflowException:
            return None

    def invalidate(self, target: str | None = None) -> None:
        with self._lock:
            if target is None:
                self._cache.clear()
            else:
                self._cache.pop(target, None)

    def status(self, target: str, refresh: bool = False) -> TargetDeployment:
        self.cfg.target(target)
        with self._lock:
            cached = self._cache.get(target)
            if cached and not refresh and time.monotonic() - cached[0] < self.CACHE_SECONDS:
                return cached[1]
        client = self._client()
        name = self.cfg.registered_model_name(target)
        deployment = TargetDeployment(
            target=target, model_name=name,
            stable_version=self._alias(client, name, self.cfg.mlflow.champion_alias),
            canary_version=self._alias(client, name, CANARY_ALIAS),
            previous_version=self._alias(client, name, PREVIOUS_ALIAS),
        )
        if deployment.canary_version:
            tags = client.get_model_version(name, deployment.canary_version).tags
            deployment.canary_stage_index = int(tags.get("canary.stage_index", 0))
            deployment.canary_percent = int(tags.get("canary.percent", self.settings.canary_stages[0]))
            deployment.canary_started_at = tags.get("canary.started_at")
            deployment.canary_stage_started_at = tags.get("canary.stage_started_at")
        with self._lock:
            self._cache[target] = (time.monotonic(), deployment)
        return deployment

    def all_status(self, targets: list[str] | None = None, refresh: bool = False) -> list[TargetDeployment]:
        return [self.status(target, refresh) for target in (targets or list(self.cfg.targets))]

    # ------------------------------------------------------------------ routing

    def route(self, target: str, routing_key: str, override: str | None = None) -> tuple[str, TargetDeployment]:
        deployment = self.status(target)
        if override == "canary" and deployment.canary_version:
            return "canary", deployment
        if override == "stable" and deployment.stable_version:
            return "stable", deployment
        if not deployment.canary_version or deployment.canary_percent <= 0:
            return "stable", deployment
        if not deployment.stable_version:
            return "canary", deployment
        bucket = traffic_bucket(target, deployment.canary_version, routing_key)
        return ("canary" if bucket < deployment.canary_percent else "stable"), deployment

    # ------------------------------------------------------------------ transitions

    def _tag(self, client, name: str, version: str, values: dict) -> None:
        for key, value in values.items():
            client.set_model_version_tag(name, version, key, str(value))

    def start_canary(self, target: str, version: str, reason: str) -> dict:
        with self._lock:
            client = self._client()
            name = self.cfg.registered_model_name(target)
            version = str(version)
            stable = self._alias(client, name, self.cfg.mlflow.champion_alias)
            if stable == version:
                return {"target": target, "action": "none", "reason": f"version {version} is already champion"}
            now = utcnow()
            if stable is None:
                # Nothing to compare against: the first model goes straight to production.
                client.set_registered_model_alias(name, self.cfg.mlflow.champion_alias, version)
                self._tag(client, name, version, {"canary.status": "promoted", "canary.promoted_at": now,
                                                  "canary.reason": f"bootstrap: {reason}"})
                self.state.add_event(target, version, "bootstrap_promote", 100, {"reason": reason})
                self.invalidate(target)
                return {"target": target, "action": "promoted", "version": version,
                        "reason": "no champion existed; first model promoted directly"}
            percent = self.settings.canary_stages[0]
            self._tag(client, name, version, {
                "canary.status": "active", "canary.stage_index": 0, "canary.percent": percent,
                "canary.started_at": now, "canary.stage_started_at": now, "canary.reason": reason,
                "canary.baseline_version": stable,
            })
            client.set_registered_model_alias(name, CANARY_ALIAS, version)
            self.state.add_event(target, version, "canary_start", percent, {"reason": reason, "stable_version": stable})
            self.invalidate(target)
            return {"target": target, "action": "canary", "version": version, "percent": percent,
                    "stable_version": stable}

    def reject(self, target: str, version: str, reason: str) -> dict:
        client = self._client()
        name = self.cfg.registered_model_name(target)
        client.set_registered_model_alias(name, self.cfg.mlflow.challenger_alias, str(version))
        self._tag(client, name, str(version), {"canary.status": "rejected", "canary.reason": reason})
        self.state.add_event(target, str(version), "rejected", 0, {"reason": reason})
        return {"target": target, "action": "rejected", "version": str(version), "reason": reason}

    def advance(self, target: str, reason: str) -> dict:
        with self._lock:
            deployment = self.status(target, refresh=True)
            if not deployment.canary_version:
                raise LookupError(f"{target} has no active canary")
            next_index = deployment.canary_stage_index + 1
            if next_index >= len(self.settings.canary_stages):
                return self.promote(target, reason)
            percent = self.settings.canary_stages[next_index]
            self._tag(self._client(), deployment.model_name, deployment.canary_version, {
                "canary.stage_index": next_index, "canary.percent": percent, "canary.stage_started_at": utcnow(),
            })
            self.state.add_event(target, deployment.canary_version, "canary_advance", percent, {"reason": reason})
            self.invalidate(target)
            return {"target": target, "action": "advanced", "version": deployment.canary_version, "percent": percent}

    def promote(self, target: str, reason: str) -> dict:
        with self._lock:
            deployment = self.status(target, refresh=True)
            if not deployment.canary_version:
                raise LookupError(f"{target} has no active canary to promote")
            client = self._client()
            name, version = deployment.model_name, deployment.canary_version
            if deployment.stable_version:
                client.set_registered_model_alias(name, PREVIOUS_ALIAS, deployment.stable_version)
            client.set_registered_model_alias(name, self.cfg.mlflow.champion_alias, version)
            client.delete_registered_model_alias(name, CANARY_ALIAS)
            self._tag(client, name, version, {"canary.status": "promoted", "canary.percent": 100,
                                              "canary.promoted_at": utcnow()})
            self.state.add_event(target, version, "promote", 100,
                                 {"reason": reason, "previous_version": deployment.stable_version})
            self.invalidate(target)
            return {"target": target, "action": "promoted", "version": version,
                    "previous_version": deployment.stable_version}

    def rollback(self, target: str, reason: str) -> dict:
        with self._lock:
            deployment = self.status(target, refresh=True)
            if not deployment.canary_version:
                raise LookupError(f"{target} has no active canary to roll back")
            client = self._client()
            name, version = deployment.model_name, deployment.canary_version
            client.delete_registered_model_alias(name, CANARY_ALIAS)
            self._tag(client, name, version, {"canary.status": "rolled_back", "canary.percent": 0,
                                              "canary.rolled_back_at": utcnow(), "canary.reason": reason})
            self.state.add_event(target, version, "rollback", 0, {"reason": reason})
            self.invalidate(target)
            return {"target": target, "action": "rolled_back", "version": version,
                    "stable_version": deployment.stable_version}

    def revert(self, target: str, reason: str) -> dict:
        """Put the previous champion back in production (the current champion becomes previous)."""
        with self._lock:
            deployment = self.status(target, refresh=True)
            if not deployment.previous_version:
                raise LookupError(f"{target} has no previous champion to revert to")
            client = self._client()
            name = deployment.model_name
            client.set_registered_model_alias(name, self.cfg.mlflow.champion_alias, deployment.previous_version)
            if deployment.stable_version:
                client.set_registered_model_alias(name, PREVIOUS_ALIAS, deployment.stable_version)
            self.state.add_event(target, deployment.previous_version, "revert", 100,
                                 {"reason": reason, "replaced_version": deployment.stable_version})
            self.invalidate(target)
            return {"target": target, "action": "reverted", "version": deployment.previous_version,
                    "replaced_version": deployment.stable_version}
