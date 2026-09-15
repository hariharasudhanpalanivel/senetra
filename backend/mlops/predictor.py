"""Forecast serving with canary traffic splitting and a prediction log for rollout decisions."""

from __future__ import annotations

import threading
import time

from senetra_ml.config import PipelineConfig
from senetra_ml.data.repository import SenetraRepository
from senetra_ml.inference import ForecastService, Scenario
from senetra_ml.tracking import ModelUnavailableError, RegistryModelLoader

from mlops.deployment import CANARY_ALIAS, DeploymentManager
from mlops.state import StateStore


class CanaryPredictor:
    def __init__(self, cfg: PipelineConfig, state: StateStore, deployments: DeploymentManager):
        self.cfg = cfg
        self.state = state
        self.deployments = deployments
        repo = SenetraRepository(cfg.data.db_path)
        # Both variants share feature data, so latency comparisons measure the models, not cache warm-up.
        panel_cache, lock = {}, threading.RLock()
        self.services = {
            "stable": ForecastService(cfg, repo, RegistryModelLoader(cfg, alias=cfg.mlflow.champion_alias),
                                      panel_cache=panel_cache, lock=lock),
            "canary": ForecastService(cfg, repo, RegistryModelLoader(cfg, alias=CANARY_ALIAS),
                                      panel_cache=panel_cache, lock=lock),
        }

    @property
    def hierarchy(self):
        return self.services["stable"].hierarchy

    def _service_for(self, variant: str, target: str, expected_version: str | None) -> ForecastService:
        service = self.services[variant]
        if expected_version and service.model(target).version != expected_version:
            service.evict_model(target)  # the alias moved (advance, promotion, rollback)
        return service

    def forecast(self, target: str, level: str, entity_id: int | None, scenario: Scenario, origin: str | None,
                 routing_key: str, override: str | None = None) -> dict:
        variant, deployment = self.deployments.route(target, routing_key, override)
        expected = deployment.canary_version if variant == "canary" else deployment.stable_version
        if expected is None:
            raise ModelUnavailableError(f"{target} has no deployed model; train and evaluate first")
        status, version, latency_ms = "ok", expected, None
        try:
            service = self._service_for(variant, target, expected)
            version = service.model(target).version  # one-time model loading is not request latency
            started = time.perf_counter()
            result = service.forecast(target, level, entity_id, scenario, origin)
            latency_ms = (time.perf_counter() - started) * 1000
        except (LookupError, ValueError):
            status = "client_error"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            self.state.log_prediction(
                target=target, variant=variant, model_version=version, level=level, entity_id=entity_id,
                scenario=scenario.name, status=status, latency_ms=latency_ms, forced=override is not None,
            )
        result["deployment"] = {
            "variant": variant, "model_version": version, "stable_version": deployment.stable_version,
            "canary_version": deployment.canary_version, "canary_percent": deployment.canary_percent,
        }
        return result

    def forecast_all(self, level: str, entity_id: int | None, scenario: Scenario, origin: str | None,
                     routing_key: str, override: str | None = None) -> dict:
        forecasts, errors = {}, {}
        for target in self.cfg.targets:
            try:
                forecasts[target] = self.forecast(target, level, entity_id, scenario, origin, routing_key, override)
            except (LookupError, ValueError):
                raise
            except Exception as exc:
                errors[target] = str(exc)
        return {"forecasts": forecasts, "errors": errors}
