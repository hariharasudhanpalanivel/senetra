"""Forecast error metrics at every level of the hierarchy.

All metrics derive from additive statistics (n, sum |e|, sum e^2, sum e, sum y), so PHC-level
numbers can be computed on each client and summed by the aggregator without sharing rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEVEL_ENTITY = {"phc": "phc_id", "district": "district_id", "country": "country_id", "world": None}
BASE_KEYS = ("fold", "origin_date", "horizon", "target_date")
FLAG_COLUMNS = ("outbreak", "transition", "cold_start")


def aggregate_to_level(frame: pd.DataFrame, level: str, value_cols: list[str], aggregation: str,
                       keys: tuple[str, ...] = BASE_KEYS) -> pd.DataFrame:
    """Bottom-up reconciliation: sum (counts) or mean (percentages) of PHC rows per entity-day."""
    entity = LEVEL_ENTITY[level]
    data = frame.dropna(subset=value_cols)
    if level == "phc":
        out = data.copy()
        out["entity_id"] = out["phc_id"]
        out["n_phcs"] = 1
        return out
    group = [k for k in keys if k in data.columns] + ([entity] if entity else [])
    spec = {c: (c, aggregation) for c in value_cols}
    spec.update({flag: (flag, "mean") for flag in FLAG_COLUMNS if flag in data.columns})
    spec["n_phcs"] = ("phc_id", "size")
    out = data.groupby(group, sort=False, observed=True).agg(**spec).reset_index()
    for flag in FLAG_COLUMNS:
        if flag in out.columns:
            out[flag] = out[flag] >= 0.5
    out["entity_id"] = out[entity] if entity else 0
    return out


def summarize_errors(frame: pd.DataFrame, models: list[str], by: list[str] | None = None,
                     actual: str = "actual") -> pd.DataFrame:
    """Tidy table: one row per (by..., model) with n, MAE, RMSE, WAPE and bias."""
    by = list(by or [])
    y = frame[actual].to_numpy(dtype=float)
    pieces = []
    for model in models:
        pred = frame[model].to_numpy(dtype=float)
        ok = np.isfinite(pred) & np.isfinite(y)
        err = pred[ok] - y[ok]
        stats = pd.DataFrame({"abs": np.abs(err), "sq": err * err, "err": err, "y": y[ok]})
        for column in by:
            stats[column] = frame[column].to_numpy()[ok]
        if by:
            sums = stats.groupby(by, observed=True).agg(
                n=("abs", "size"), abs=("abs", "sum"), sq=("sq", "sum"), err=("err", "sum"), y=("y", "sum")
            ).reset_index()
        else:
            sums = pd.DataFrame([{"n": len(stats), "abs": stats["abs"].sum(), "sq": stats["sq"].sum(),
                                  "err": stats["err"].sum(), "y": stats["y"].sum()}])
        sums.insert(0, "model", model)
        pieces.append(sums)
    table = pd.concat(pieces, ignore_index=True)
    n = table["n"].clip(lower=1)
    denom = table["y"].where(table["y"].abs() > 1e-12)
    table["mae"] = table["abs"] / n
    table["rmse"] = np.sqrt(table["sq"] / n)
    table["wape"] = table["abs"] / denom
    table["bias"] = table["err"] / denom
    return table.drop(columns=["abs", "sq", "err", "y"])


def skill(model_error: float, baseline_error: float) -> float:
    """1 - model / baseline; positive means the model beats the baseline."""
    if not np.isfinite(baseline_error) or baseline_error <= 0:
        return float("nan")
    return float(1.0 - model_error / baseline_error)
