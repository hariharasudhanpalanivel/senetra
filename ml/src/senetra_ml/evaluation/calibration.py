"""Model-level selection, conformal prediction intervals and monitoring reference metrics.

Intervals are split-conformal on relative residuals r = (y - yhat) / max(|yhat|, 1), estimated
per hierarchy level and horizon from backtest residuals. Two interval kinds are stored:
- "scenario": residuals when the scenario covariate is correct (what-if / simulated forecasts);
- "observed": residuals when the current outbreak state is assumed to persist (operational).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from senetra_ml.config import LEVELS
from senetra_ml.evaluation.metrics import aggregate_to_level, summarize_errors

QUANTILES = {"lo80": 0.10, "hi80": 0.90, "lo95": 0.025, "hi95": 0.975}
INTERVAL_KINDS = {"scenario": "oracle", "observed": "observed"}


def usable_folds(frame: pd.DataFrame) -> pd.DataFrame:
    """Folds whose training window contained an outbreak; cold-start folds are reported separately."""
    warm = frame[~frame["cold_start"]]
    return warm if len(warm) else frame


def _scale(pred: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(pred), 1.0)


def quantile_table(actual: np.ndarray, pred: np.ndarray, horizon: np.ndarray) -> dict:
    ok = np.isfinite(actual) & np.isfinite(pred)
    actual, pred, horizon = actual[ok], pred[ok], horizon[ok]
    if actual.size == 0:
        return {}
    residual = (actual - pred) / _scale(pred)
    table = {"all": {k: float(np.quantile(residual, q)) for k, q in QUANTILES.items()}}
    for h in np.unique(horizon):
        r = residual[horizon == h]
        if r.size >= 20:
            table[str(int(h))] = {k: float(np.quantile(r, q)) for k, q in QUANTILES.items()}
    return table


def interval_bounds(pred: np.ndarray, horizon: np.ndarray, table: dict) -> dict[str, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    if not table:
        return {k: pred.copy() for k in QUANTILES}
    rows = [table.get(str(int(h)), table["all"]) for h in horizon]
    scale = _scale(pred)
    return {k: pred + np.array([row[k] for row in rows]) * scale for k in QUANTILES}


def select_level(frame: pd.DataFrame) -> tuple[str, dict[str, float]]:
    """Pick the federation level whose parameters serve PHC forecasts, by backtest PHC MAE."""
    data = usable_folds(frame)
    scores = {level: float(np.nanmean(np.abs(data[f"fl_{level}__observed"] - data["actual"])))
              for level in LEVELS}
    return min(scores, key=scores.get), scores


def conformal_tables(frame: pd.DataFrame, variant: str, aggregation: str) -> dict:
    data = usable_folds(frame)
    columns = ["actual"] + [f"{variant}__{suffix}" for suffix in INTERVAL_KINDS.values()]
    tables: dict = {kind: {} for kind in INTERVAL_KINDS}
    for level in LEVELS:
        agg = aggregate_to_level(data, level, columns, aggregation)
        for kind, suffix in INTERVAL_KINDS.items():
            tables[kind][level] = quantile_table(agg["actual"].to_numpy(float),
                                                 agg[f"{variant}__{suffix}"].to_numpy(float),
                                                 agg["horizon"].to_numpy(int))
    return tables


def cross_fitted_coverage(frame: pd.DataFrame, variant: str, aggregation: str, level: str,
                          kind: str = "observed") -> dict:
    """Interval coverage where each fold is calibrated on the other folds only."""
    data = usable_folds(frame)
    column = f"{variant}__{INTERVAL_KINDS[kind]}"
    agg = aggregate_to_level(data, level, ["actual", column], aggregation)
    folds = agg["fold"].unique()
    if len(folds) < 2:
        return {"coverage80": float("nan"), "coverage95": float("nan"), "cross_fitted": False}
    hits80, hits95, total = 0, 0, 0
    for fold in folds:
        calib, test = agg[agg["fold"] != fold], agg[agg["fold"] == fold]
        table = quantile_table(calib["actual"].to_numpy(float), calib[column].to_numpy(float),
                               calib["horizon"].to_numpy(int))
        bounds = interval_bounds(test[column].to_numpy(float), test["horizon"].to_numpy(int), table)
        y = test["actual"].to_numpy(float)
        hits80 += int(((y >= bounds["lo80"]) & (y <= bounds["hi80"])).sum())
        hits95 += int(((y >= bounds["lo95"]) & (y <= bounds["hi95"])).sum())
        total += len(test)
    return {"coverage80": hits80 / max(total, 1), "coverage95": hits95 / max(total, 1), "cross_fitted": True}


def reference_metrics(frame: pd.DataFrame, variant: str, aggregation: str) -> dict:
    """Operational (observed-scenario) backtest errors that monitoring compares live errors against."""
    data = usable_folds(frame)
    column = f"{variant}__observed"
    out = {}
    for level in ("phc", "district"):
        agg = aggregate_to_level(data, level, ["actual", column], aggregation)
        row = summarize_errors(agg, [column]).iloc[0]
        out[f"{level}_mae"] = float(row["mae"])
        out[f"{level}_wape"] = float(row["wape"])
    return out
