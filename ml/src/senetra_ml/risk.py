"""Transparent stock-out risk rules applied on top of demand forecasts."""

from __future__ import annotations

import math

import numpy as np

from senetra_ml.config import RiskConfig


def days_of_supply(stock: float | None, daily_forecast) -> float | None:
    """Days until cumulative forecast demand exhausts the stock (extrapolated past the horizon)."""
    if stock is None or not math.isfinite(stock):
        return None
    daily = np.clip(np.asarray(daily_forecast, dtype=float), 0.0, None)
    if stock <= 0:
        return 0.0
    if daily.size == 0:
        return None
    cumulative = np.cumsum(daily)
    idx = int(np.searchsorted(cumulative, stock, side="left"))
    if idx < daily.size:
        before = cumulative[idx - 1] if idx > 0 else 0.0
        return float(idx + (stock - before) / daily[idx]) if daily[idx] > 0 else float(idx)
    mean = float(daily.mean())
    return math.inf if mean <= 0 else float(daily.size + (stock - cumulative[-1]) / mean)


def risk_level(days: float | None, cfg: RiskConfig) -> str:
    if days is None:
        return "UNKNOWN"
    if days <= cfg.critical_days:
        return "CRITICAL"
    if days <= cfg.high_days:
        return "HIGH"
    if days <= cfg.watch_days:
        return "WATCH"
    return "HEALTHY"
