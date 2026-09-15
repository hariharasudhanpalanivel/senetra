"""Training pipeline.

Per target: build features in memory -> rolling-origin backtest of the hierarchical federated GLM
and its benchmarks -> choose the serving federation level -> fit on all history -> conformal
calibration -> log and register one MLflow model. Backtest predictions are logged without
actuals; evaluation re-reads actuals from the database.
"""

from __future__ import annotations

import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np

from senetra_ml import __version__
from senetra_ml.config import PipelineConfig
from senetra_ml.data.db import database_fingerprint
from senetra_ml.data.panel import Panel, load_panel
from senetra_ml.data.repository import SenetraRepository, day
from senetra_ml.data.validation import validate_database
from senetra_ml.evaluation.calibration import conformal_tables, reference_metrics, select_level
from senetra_ml.features.engineering import build_features
from senetra_ml.inference import model_input_frame
from senetra_ml.models.bundle import ForecastBundle
from senetra_ml.models.pyfunc import log_forecast_model
from senetra_ml.monitoring.drift import build_reference_profile
from senetra_ml.tracking import experiment_id, lineage_tags, setup_mlflow
from senetra_ml.training.backtest import fit_models, run_backtest

log = logging.getLogger(__name__)

PRIVACY_CONTRACT = [
    "Row-level PHC records are read only by that PHC's own client (PHC-scoped queries).",
    "PHC clients share summed loss/gradient/Hessian statistics (or parameter deltas with FedAvg), "
    "training-row counts and feature moments, never rows.",
    "Districts also receive daily outbreak-flag counts (surveillance prevalence).",
    "Country and world tiers receive model parameters and row counts only.",
]


def _hierarchy_summary(panel: Panel) -> list[dict]:
    return [
        {"country_id": int(cid), "country": str(group["country_name"].iloc[0]),
         "districts": int(group["district_id"].nunique()), "phcs": int(len(group))}
        for cid, group in panel.hierarchy.groupby("country_id")
    ]


def _outbreak_effect(world: np.ndarray, transform, family: str) -> float:
    """Effect of moving scenario intensity from 0 to 1 with other inputs fixed (x multiplier or +units)."""
    beta = world[1 + transform.scenario_index] / transform.scale[transform.scenario_index]
    return float(np.exp(beta)) if family == "poisson" else float(beta)


def train_target(cfg: PipelineConfig, panel: Panel, target: str, exp_id: str, register: bool) -> dict:
    target_cfg = cfg.target(target)
    started = time.perf_counter()
    with mlflow.start_run(experiment_id=exp_id, run_name=target, nested=True) as run:
        mlflow.set_tags({"senetra.stage": "training", "senetra.role": "target", "senetra.target": target})
        fs = build_features(panel, target, cfg)
        mlflow.log_params({"target": target, "family": target_cfg.family, "aggregation": target_cfg.aggregation,
                           "n_features": len(fs.names), "features": ",".join(fs.names)})
        log.info("[%s] backtesting: %d PHCs x %d origin-horizon rows", target, *fs.y.shape)
        frame, folds = run_backtest(
            fs, panel, cfg,
            progress=lambda s: log.info("[%s] fold %d cutoff=%s cold_start=%s test_rows=%d",
                                        target, s["fold"], s["cutoff_date"], s["cold_start"], s["test_rows"]),
        )
        selected, level_scores = select_level(frame)
        variant = f"fl_{selected}"
        conformal = conformal_tables(frame, variant, target_cfg.aggregation)
        reference = reference_metrics(frame, variant, target_cfg.aggregation)
        mlflow.log_metrics({f"backtest_phc_mae_fl_{level}": score for level, score in level_scores.items()})
        mlflow.log_metrics({f"reference_{key}": value for key, value in reference.items()})
        for summary in folds:
            for point in summary["federated_history"]:
                mlflow.log_metric(f"fold{summary['fold']}_world_objective", point["world_objective"],
                                  step=point["round"])

        last = panel.n_dates - 1
        history_ok = fs.history_ok()
        final = fit_models(fs, panel, last, cfg, history_ok, include_benchmarks=False)
        fed = final.federated
        for point in fed.history:
            mlflow.log_metric("final_world_objective", point["world_objective"], step=point["round"])
        train_mask = fs.train_mask(last, history_ok)

        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_version": __version__,
            "data_fingerprint": database_fingerprint(cfg.data.db_path),
            "training_start": day(panel.dates[0]),
            "training_end": day(panel.dates[-1]),
            "train_rows": final.train_rows,
            "outbreak_threshold": cfg.data.outbreak_prevalence_threshold,
            "federated_algorithm": cfg.federated.algorithm,
            "outbreak_effect": _outbreak_effect(fed.world, final.transform, target_cfg.family),
            "world_coefficients": dict(zip(["intercept", *fs.names], fed.world.tolist())),
            "hierarchy": _hierarchy_summary(panel),
            "n_phcs": int(panel.n_phcs), "n_districts": int(len(panel.district_ids)),
            "n_countries": int(len(panel.country_ids)),
            "federated_config": cfg.federated.model_dump(),
            "federated_history": fed.history,
            "level_phc_mae": level_scores,
            "reference_metrics": reference,
            "backtest_folds": [{k: v for k, v in f.items() if k != "federated_history"} for f in folds],
            "privacy_contract": PRIVACY_CONTRACT,
        }
        bundle = ForecastBundle(
            target=target, target_config=target_cfg.model_dump(mode="json"), transform=final.transform,
            selected_level=selected, world=fed.world, country_ids=fed.country_ids, country=fed.country,
            district_ids=fed.district_ids, district=fed.district, phc_ids=fed.phc_ids, phc=fed.phc,
            conformal=conformal, metadata=metadata,
            reference_profile=build_reference_profile(fs, train_mask, final.transform),
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            predictions = frame.drop(columns=["actual"])
            floats = predictions.select_dtypes("float64").columns
            predictions[floats] = predictions[floats].astype("float32")
            predictions.to_parquet(tmp_dir / "predictions.parquet", index=False)
            mlflow.log_artifact(str(tmp_dir / "predictions.parquet"), artifact_path="backtest")
            mlflow.log_dict({"folds": folds, "level_phc_mae": level_scores, "reference_metrics": reference},
                            "backtest/summary.json")

            example_mask = np.zeros(fs.y.shape, dtype=bool)
            example_mask[: min(2, panel.n_phcs), fs.origin_idx == last - cfg.forecast.horizon_days] = True
            input_example = model_input_frame(fs, panel, example_mask)
            bundle_dir = bundle.save(tmp_dir / "bundle")
            info = log_forecast_model(bundle, bundle_dir, input_example,
                                      cfg.registered_model_name(target) if register else None)

        version = getattr(info, "registered_model_version", None)
        seconds = time.perf_counter() - started
        mlflow.set_tags({"senetra.selected_level": selected,
                         "senetra.model_version": str(version) if version else "unregistered",
                         "senetra.model_uri": info.model_uri})
        mlflow.log_metric("training_seconds", seconds)
        log.info("[%s] done in %.0fs: level=%s version=%s district WAPE (observed)=%.4f",
                 target, seconds, selected, version, reference["district_wape"])
        return {"run_id": run.info.run_id, "model_version": version, "selected_level": selected,
                "reference_metrics": reference, "outbreak_effect": metadata["outbreak_effect"],
                "seconds": round(seconds, 1)}


def run_training(cfg: PipelineConfig, targets: list[str] | None = None, register: bool = True) -> dict:
    targets = cfg.resolve_targets(targets)
    setup_mlflow(cfg)
    with SenetraRepository(cfg.data.db_path) as repo:
        report = validate_database(repo, cfg)
        for check in report.warnings:
            log.warning("data validation: %s: %s", check.name, check.detail)
        report.raise_for_errors()
        panel = load_panel(repo)
        log.info("Panel: %d PHCs, %d districts, %d countries, %d days", panel.n_phcs,
                 len(panel.district_ids), len(panel.country_ids), panel.n_dates)
        exp_id = experiment_id(cfg, "training")
        with mlflow.start_run(experiment_id=exp_id,
                              run_name=f"train-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}") as parent:
            mlflow.set_tags({**lineage_tags(cfg, repo), "senetra.stage": "training", "senetra.role": "parent"})
            mlflow.log_dict(cfg.model_dump(mode="json"), "config/pipeline.json")
            mlflow.log_dict(report.to_dict(), "data_validation.json")
            mlflow.log_params({
                "targets": ",".join(targets), "horizon_days": cfg.forecast.horizon_days,
                "fl_rounds": cfg.federated.rounds, "fl_district_rounds": cfg.federated.district_rounds,
                "backtest_folds": cfg.backtest.n_folds, "n_phcs": panel.n_phcs,
                "n_districts": len(panel.district_ids), "n_countries": len(panel.country_ids),
                "n_days": panel.n_dates,
            })
            results = {target: train_target(cfg, panel, target, exp_id, register) for target in targets}
            mlflow.log_dict(results, "training_summary.json")
    return {"run_id": parent.info.run_id, "targets": results}
