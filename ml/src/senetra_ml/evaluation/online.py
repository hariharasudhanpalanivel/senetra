"""Replay registered models over realized data.

Used by monitoring (live performance of the champion) and by canary deployments (champion vs canary
on actuals that arrived after both models were trained). Scenario: observed persistence, i.e. the
same information the model would have had in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from senetra_ml.config import PipelineConfig
from senetra_ml.data.panel import Panel, load_panel
from senetra_ml.data.repository import SenetraRepository, day
from senetra_ml.features.engineering import SCENARIO_FEATURE, FeatureSet, build_features
from senetra_ml.inference import model_input_frame
from senetra_ml.models.bundle import ForecastBundle


def replay_performance(fs: FeatureSet, panel: Panel, bundle: ForecastBundle, row_mask: np.ndarray,
                       aggregation: str) -> dict:
    """PHC MAE and district WAPE of `bundle` on the rows in `row_mask` (targets must be known)."""
    if not row_mask.any():
        return {"rows": 0}
    frame = model_input_frame(fs, panel, row_mask)
    frame[SCENARIO_FEATURE] = np.nan_to_num(fs.origin_outbreak()[row_mask])
    frame["yhat"], _, _ = bundle.predict_rows(frame)
    frame["actual"] = fs.y[row_mask]
    district = frame.groupby(["district_id", "origin_date", "horizon"]).agg(
        yhat=("yhat", aggregation), actual=("actual", aggregation))
    return {
        "rows": int(len(frame)),
        "phc_mae": float(np.mean(np.abs(frame["yhat"] - frame["actual"]))),
        "district_wape": float(np.abs(district["yhat"] - district["actual"]).sum() / district["actual"].sum()),
        "first_target_date": str(frame["target_date"].min()),
        "last_target_date": str(frame["target_date"].max()),
    }


def fresh_window_panel(cfg: PipelineConfig, repo: SenetraRepository, after_date) -> Panel | None:
    """Panel covering target days after `after_date` plus the history their features need."""
    lo, hi = repo.date_bounds()
    first = pd.Timestamp(after_date).normalize() + pd.Timedelta(days=1)
    if first > hi:
        return None
    lookback = cfg.forecast.long_window + cfg.data.min_history_days + cfg.forecast.horizon_days + 7
    return load_panel(repo, max(lo, first - pd.Timedelta(days=lookback)), hi)


def compare_on_fresh_data(cfg: PipelineConfig, panel: Panel | None, target: str,
                          bundles: dict[str, ForecastBundle], after_date) -> dict:
    """Errors of each model on target days after `after_date`, which none of them was trained on."""
    first = pd.Timestamp(after_date).normalize() + pd.Timedelta(days=1)
    pending = {"status": "pending", "reason": f"no actuals after {day(after_date)} yet"}
    if panel is None or first > panel.dates[-1]:
        return pending
    fs = build_features(panel, target, cfg)
    first_idx = panel.date_index(first) if first >= panel.dates[0] else 0
    mask = fs.history_ok() & np.isfinite(fs.y) & (fs.target_idx >= first_idx)[None, :]
    if not mask.any():
        return pending
    aggregation = cfg.target(target).aggregation
    return {
        "status": "compared",
        "after_date": day(after_date),
        "models": {name: replay_performance(fs, panel, bundle, mask, aggregation) for name, bundle in bundles.items()},
    }
