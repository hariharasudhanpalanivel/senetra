"""Monitoring pipeline: data quality, regime-matched feature/target drift and live performance.

Performance is measured by replaying the champion over the most recent window with the
operational (observed) scenario and comparing errors to the backtest reference metrics stored
with the model. Status: ok | warn | alert; `--fail-on-alert` turns alerts into a non-zero exit.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd

from senetra_ml.config import PipelineConfig
from senetra_ml.data.panel import Panel, load_panel
from senetra_ml.data.repository import SenetraRepository, day, next_day
from senetra_ml.data.validation import validate_database
from senetra_ml.evaluation.online import replay_performance
from senetra_ml.features.engineering import build_features
from senetra_ml.monitoring.drift import TARGET_KEY, compare_to_profile, regime_samples
from senetra_ml.tracking import ModelUnavailableError, RegistryModelLoader, experiment_id, setup_mlflow

log = logging.getLogger(__name__)


def _alert(alerts: list, scope: str, check: str, severity: str, detail: str) -> None:
    alerts.append({"scope": scope, "check": check, "severity": severity, "detail": detail})


def monitor_target(cfg: PipelineConfig, panel: Panel, loader: RegistryModelLoader, target: str,
                   window: int, alerts: list) -> dict:
    mon = cfg.monitoring
    try:
        loaded = loader.load(target)
    except ModelUnavailableError as exc:
        _alert(alerts, target, "model_available", "alert", str(exc))
        return {"status": "no_model"}
    bundle = loaded.bundle
    target_cfg = cfg.target(target)
    fs = build_features(panel, target, cfg)
    last = panel.n_dates - 1
    history_ok = fs.history_ok()
    known = np.isfinite(fs.y) & np.isfinite(fs.X[..., fs.scenario_index])
    recent_origins = (fs.origin_idx >= last - window + 1)[None, :]
    recent_targets = (fs.target_idx >= last - window + 1)[None, :]

    samples = regime_samples(fs, bundle.transform, history_ok & recent_origins,
                             history_ok & np.isfinite(fs.y) & recent_targets)
    drift = compare_to_profile(bundle.reference_profile, samples)
    recent_rows = {regime: int(values[TARGET_KEY].size) for regime, values in samples.items()}
    recent_share = recent_rows["outbreak"] / max(sum(recent_rows.values()), 1)
    reference_share = bundle.reference_profile.get("outbreak_share", 0.0)
    if abs(recent_share - reference_share) > 0.25:
        _alert(alerts, target, "regime_shift", "warn",
               f"outbreak share of recent rows {recent_share:.0%} vs {reference_share:.0%} in training")
    max_psi = 0.0
    for regime, features in drift.items():
        for name, result in features.items():
            value = result["psi"]
            if not np.isfinite(value):
                continue
            max_psi = max(max_psi, value)
            label = "target" if name == TARGET_KEY else name
            if value >= mon.psi_alert:
                severity = "warn" if result["regime_feature"] or result["phase_dependent"] else "alert"
                _alert(alerts, target, f"drift:{regime}:{label}", severity, f"PSI {value:.3f} >= {mon.psi_alert}")
            elif value >= mon.psi_warn:
                _alert(alerts, target, f"drift:{regime}:{label}", "warn", f"PSI {value:.3f} >= {mon.psi_warn}")

    perf_mask = history_ok & known & recent_targets
    performance = {}
    if perf_mask.any():
        replay = replay_performance(fs, panel, bundle, perf_mask, target_cfg.aggregation)
        phc_mae, district_wape = replay["phc_mae"], replay["district_wape"]
        reference = bundle.metadata.get("reference_metrics", {})
        training_end = pd.Timestamp(bundle.metadata.get("training_end", "1900-01-01"))
        target_days = fs.start_date + pd.to_timedelta(
            np.broadcast_to(fs.target_idx, perf_mask.shape)[perf_mask], unit="D")
        in_sample = float((target_days <= training_end).mean())
        performance = {
            "rows": replay["rows"], "phc_mae": phc_mae, "district_wape": district_wape,
            "phc_mae_ratio": phc_mae / reference["phc_mae"] if reference.get("phc_mae") else None,
            "district_wape_ratio": district_wape / reference["district_wape"] if reference.get("district_wape") else None,
            "in_sample_share": in_sample,
        }
        for key in ("phc_mae_ratio", "district_wape_ratio"):
            ratio = performance[key]
            if ratio is not None and ratio > mon.degradation_ratio_alert:
                _alert(alerts, target, f"performance:{key}", "alert",
                       f"{ratio:.2f}x the backtest reference (limit {mon.degradation_ratio_alert}x)")
        if in_sample > 0.5:
            _alert(alerts, target, "performance:in_sample", "warn",
                   f"{in_sample:.0%} of the window was used for training, so errors are optimistic; "
                   "monitor again once new data arrives")
    return {
        "status": "checked", "model": loaded.label, "max_psi": max_psi, "recent_outbreak_share": recent_share,
        "drift": drift, "performance": performance,
    }


def run_monitoring(cfg: PipelineConfig, targets: list[str] | None = None, window_days: int | None = None,
                   today: pd.Timestamp | None = None) -> dict:
    targets = cfg.resolve_targets(targets)
    window = window_days or cfg.monitoring.window_days
    setup_mlflow(cfg)
    loader = RegistryModelLoader(cfg)
    alerts: list[dict] = []
    with SenetraRepository(cfg.data.db_path) as repo:
        validation = validate_database(repo, cfg, today)
        for check in validation.checks:
            if check.status in ("warn", "error"):
                _alert(alerts, "data", check.name, "alert" if check.status == "error" else "warn", check.detail)
        lo, hi = repo.date_bounds()
        (reporting,), = repo.rows("SELECT COUNT(DISTINCT phc_id) FROM daily_metrics WHERE date >= ? AND date < ?",
                                  (day(hi), next_day(hi)))
        (total_phcs,), = repo.rows("SELECT COUNT(*) FROM phcs")
        reporting_ratio = reporting / max(total_phcs, 1)
        if reporting_ratio < cfg.monitoring.min_reporting_ratio:
            _alert(alerts, "data", "reporting_ratio", "alert",
                   f"only {reporting_ratio:.1%} of PHCs reported on {day(hi)}")
        lookback = window + cfg.forecast.horizon_days + cfg.forecast.long_window + cfg.data.min_history_days + 7
        panel = load_panel(repo, hi - pd.Timedelta(days=lookback), hi)
        per_target = {target: monitor_target(cfg, panel, loader, target, window, alerts) for target in targets}

    severities = {a["severity"] for a in alerts}
    status = "alert" if "alert" in severities else "warn" if "warn" in severities else "ok"
    stamp = datetime.now(timezone.utc)
    report = {
        "generated_at": stamp.isoformat(timespec="seconds"), "status": status, "window_days": window,
        "data": {"latest_date": day(hi), "reporting_ratio": reporting_ratio, "validation": validation.to_dict()},
        "targets": per_target, "alerts": alerts,
    }
    out_dir = cfg.reports_dir / "monitoring"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, default=str)
    (out_dir / "latest.json").write_text(text, encoding="utf-8")
    (out_dir / f"monitoring-{stamp:%Y%m%d-%H%M%S}.json").write_text(text, encoding="utf-8")

    with mlflow.start_run(experiment_id=experiment_id(cfg, "monitoring"), run_name=f"monitor-{stamp:%Y%m%d-%H%M%S}"):
        mlflow.set_tags({"senetra.stage": "monitoring", "status": status})
        metrics = {"alerts": sum(a["severity"] == "alert" for a in alerts),
                   "warnings": sum(a["severity"] == "warn" for a in alerts), "reporting_ratio": reporting_ratio}
        for target, result in per_target.items():
            if result.get("status") != "checked":
                continue
            metrics[f"{target}.max_psi"] = result["max_psi"]
            for key in ("phc_mae", "district_wape", "phc_mae_ratio", "district_wape_ratio"):
                value = result["performance"].get(key)
                if value is not None:
                    metrics[f"{target}.{key}"] = value
        mlflow.log_metrics(metrics)
        mlflow.log_dict(report, "monitoring_report.json")
    log.info("Monitoring status: %s (%d alerts, %d warnings)", status, metrics["alerts"], metrics["warnings"])
    return report
