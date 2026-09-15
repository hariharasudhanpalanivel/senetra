"""Batch prediction pipeline: champion forecasts for a scope and scenario, logged to MLflow."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from senetra_ml.config import PipelineConfig
from senetra_ml.inference import ForecastService, Scenario
from senetra_ml.tracking import ModelUnavailableError, RegistryModelLoader, experiment_id, setup_mlflow

log = logging.getLogger(__name__)


def run_prediction(cfg: PipelineConfig, targets: list[str] | None = None, level: str = "district",
                   entity_id: int | None = None, scenario: Scenario | None = None, origin: str | None = None,
                   write_db: bool = False, output: str | Path | None = None,
                   allow_unpromoted: bool = False) -> dict:
    scenario = scenario or Scenario()
    targets = cfg.resolve_targets(targets)
    setup_mlflow(cfg)
    service = ForecastService(cfg, loader=RegistryModelLoader(cfg, allow_latest=allow_unpromoted))
    forecasts, errors, written = {}, {}, {}
    for target in targets:
        try:
            forecasts[target] = service.forecast(target, level, entity_id, scenario, origin)
        except ModelUnavailableError as exc:
            errors[target] = str(exc)
            log.error("[%s] %s", target, exc)
            continue
        if write_db and cfg.target(target).is_medicine:
            written[target] = service.write_predictions(target, scenario, origin)
            log.info("[%s] wrote %d PHC forecasts to the predictions table", target, written[target])

    stamp = datetime.now(timezone.utc)
    payload = {
        "generated_at": stamp.isoformat(timespec="seconds"), "level": level, "entity_id": entity_id,
        "scenario": scenario.to_dict(), "forecasts": forecasts, "errors": errors, "rows_written": written,
    }
    path = Path(output) if output else (
        cfg.reports_dir / "predictions" / f"forecast-{level}-{entity_id or 'all'}-{scenario.name}-{stamp:%Y%m%d-%H%M%S}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with mlflow.start_run(experiment_id=experiment_id(cfg, "predictions"),
                          run_name=f"predict-{level}-{entity_id or 'all'}-{scenario.name}"):
        mlflow.set_tags({"senetra.stage": "prediction", **{
            f"model.{t}": f"{f['model']['name']}:v{f['model']['version']}" for t, f in forecasts.items()}})
        mlflow.log_params({"level": level, "entity_id": entity_id, "scenario": scenario.name,
                           "severity": scenario.severity, "origin": origin or "latest", "write_db": write_db})
        mlflow.log_dict(payload, "forecasts.json")
    payload["output_path"] = str(path)
    return payload
