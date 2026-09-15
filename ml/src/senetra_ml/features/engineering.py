"""Leakage-safe feature construction on the in-memory panel.

A row is (PHC, origin day t, horizon h) and forecasts the target on day d = t + h.
- History features use only values observed on or before t.
- Target-day inputs are the calendar and the scenario covariate `scenario_outbreak`: whether the
  PHC's district is in an outbreak on d (prevalence of outbreak flags above the threshold).
  Training uses the realized regime (0/1); inference substitutes a scenario intensity (normal 0,
  observed persistence, simulated events, or a what-if severity scaled so 1.0 = historical level).

EDA showed consumption is near-i.i.d. within a regime and shifts as a multiplicative step during
outbreaks. A binary regime covariate matches that step; a continuous, noisy prevalence would
attenuate the learned effect and leak it into correlated origin-day features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from senetra_ml.config import PipelineConfig
from senetra_ml.data.panel import Panel

SCENARIO_FEATURE = "scenario_outbreak"
BASELINES = ("bl_naive", "bl_seasonal7", "bl_mean7")


def trailing_stats(x: np.ndarray, window: int, min_periods: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std over the `window` days ending at each t (inclusive), ignoring NaNs. x: (C, T)."""
    valid = ~np.isnan(x)
    filled = np.where(valid, x, 0.0)
    pad = np.zeros((x.shape[0], 1))
    csum = np.concatenate([pad, np.cumsum(filled, axis=1)], axis=1)
    csq = np.concatenate([pad, np.cumsum(filled * filled, axis=1)], axis=1)
    ccount = np.concatenate([pad, np.cumsum(valid, axis=1)], axis=1)
    hi = np.arange(1, x.shape[1] + 1)
    lo = np.maximum(hi - window, 0)
    count = ccount[:, hi] - ccount[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (csum[:, hi] - csum[:, lo]) / count
        var = (csq[:, hi] - csq[:, lo]) / count - mean * mean
    enough = count >= min_periods
    return np.where(enough, mean, np.nan), np.where(enough, np.sqrt(np.clip(var, 0.0, None)), np.nan)


@dataclass
class FeatureSet:
    target: str
    family: str
    names: tuple[str, ...]
    X: np.ndarray  # (C, N, F) raw features
    y: np.ndarray  # (C, N) realized target on the target day; NaN when unknown
    origin_idx: np.ndarray  # (N,) panel day index of the forecast origin
    horizon: np.ndarray  # (N,)
    target_idx: np.ndarray  # (N,) may exceed the panel for future days
    start_date: pd.Timestamp  # panel day 0
    baselines: dict[str, np.ndarray]  # (C, N) raw-scale naive forecasts
    target_prevalence: np.ndarray  # (C, N) realized district prevalence on the target day
    origin_prevalence: np.ndarray  # (C, N) district prevalence on the origin day
    outbreak_threshold: float = 0.02
    # (C, N) the district had an outbreak day inside the long history window ending at the origin / target day.
    origin_recent_outbreak: np.ndarray | None = None
    target_recent_outbreak: np.ndarray | None = None
    long_window: int = 28

    @property
    def scenario_index(self) -> int:
        return self.names.index(SCENARIO_FEATURE)

    def origin_outbreak(self) -> np.ndarray:
        """(C, N) 1.0 where the district was in an outbreak on the origin day (NaN if unreported)."""
        prevalence = self.origin_prevalence
        return np.where(np.isfinite(prevalence), (prevalence > self.outbreak_threshold).astype(float), np.nan)

    @property
    def origin_dates(self) -> pd.DatetimeIndex:
        return self.start_date + pd.to_timedelta(self.origin_idx, unit="D")

    @property
    def target_dates(self) -> pd.DatetimeIndex:
        return self.start_date + pd.to_timedelta(self.target_idx, unit="D")

    def history_ok(self) -> np.ndarray:
        """(C, N) rows whose history features are all finite (scenario column excluded)."""
        ok = np.ones(self.y.shape, dtype=bool)
        for f in range(self.X.shape[-1]):
            if f != self.scenario_index:
                ok &= np.isfinite(self.X[..., f])
        return ok

    def train_mask(self, cutoff_idx: int, history_ok: np.ndarray | None = None) -> np.ndarray:
        """Rows usable for training when only targets on or before `cutoff_idx` are known."""
        ok = self.history_ok() if history_ok is None else history_ok
        return (ok & np.isfinite(self.y) & np.isfinite(self.X[..., self.scenario_index])
                & (self.target_idx <= cutoff_idx)[None, :])


def build_features(panel: Panel, target_name: str, cfg: PipelineConfig,
                   origins: np.ndarray | None = None) -> FeatureSet:
    target = cfg.target(target_name)
    horizon_days = cfg.forecast.horizon_days
    n_dates = panel.n_dates
    seasonal_lag = 7 * int(np.ceil(horizon_days / 7))
    first_origin = max(cfg.data.min_history_days, seasonal_lag) - 1
    origins = np.arange(first_origin, n_dates) if origins is None else np.asarray(origins, dtype=int)
    if origins.size == 0 or origins.min() < first_origin or origins.max() >= n_dates:
        raise ValueError(f"Origins must lie in [{first_origin}, {n_dates - 1}] of the panel")

    origin_idx = np.repeat(origins, horizon_days)
    horizon = np.tile(np.arange(1, horizon_days + 1), origins.size)
    target_idx = origin_idx + horizon

    poisson = target.family == "poisson"
    level = (lambda a: np.log1p(np.clip(a, 0.0, None))) if poisson else (lambda a: a)
    series = panel.series(target.series)
    long_window = cfg.forecast.long_window

    history: dict[str, np.ndarray] = {"y_last": level(series)}
    stats = {w: trailing_stats(series, w, max(1, w // 2)) for w in cfg.forecast.rolling_windows}
    for w, (mean, _) in stats.items():
        history[f"y_mean{w}"] = level(mean)
    mean7, std7 = stats[7]
    mean_long = stats[long_window][0]
    history["y_disp7"] = std7 / (mean7 + 1.0) if poisson else std7
    history["y_trend"] = np.log((mean7 + 1.0) / (mean_long + 1.0)) if poisson else mean7 - mean_long

    if target.series != "patient_footfall":
        footfall = panel.series("patient_footfall")
        ff7 = trailing_stats(footfall, 7, 4)[0]
        ff_long = trailing_stats(footfall, long_window, long_window // 2)[0]
        history["ff_mean7"] = np.log1p(ff7)
        history["ff_trend"] = np.log((ff7 + 1.0) / (ff_long + 1.0))
    if target.series != "bed_occupancy":
        history["bed_mean7"] = trailing_stats(panel.series("bed_occupancy"), 7, 4)[0]
    history["flag_share7"] = trailing_stats(panel.series("outbreak_flag"), 7, 4)[0]

    prevalence = panel.client_prevalence()
    history["dprev_origin"] = prevalence
    history["dprev_mean7"] = trailing_stats(prevalence, 7, 4)[0]

    in_range = target_idx < n_dates
    safe_target = np.minimum(target_idx, n_dates - 1)
    target_prevalence = np.where(in_range[None, :], prevalence[:, safe_target], np.nan)
    y = np.where(in_range[None, :], series[:, safe_target], np.nan)

    start_date = panel.dates[0]
    dow = (start_date + pd.to_timedelta(target_idx, unit="D")).dayofweek.to_numpy()
    n_phcs = panel.n_phcs
    threshold = cfg.data.outbreak_prevalence_threshold
    outbreak_day = np.where(np.isfinite(prevalence), (prevalence > threshold).astype(float), np.nan)
    recent_outbreak = trailing_stats(outbreak_day, long_window, 1)[0] > 0
    target_day = {
        SCENARIO_FEATURE: np.where(np.isfinite(target_prevalence), (target_prevalence > threshold).astype(float), np.nan),
        "dow_sin": np.broadcast_to(np.sin(2 * np.pi * dow / 7), (n_phcs, dow.size)),
        "dow_cos": np.broadcast_to(np.cos(2 * np.pi * dow / 7), (n_phcs, dow.size)),
        "horizon": np.broadcast_to(horizon.astype(float), (n_phcs, horizon.size)),
    }

    names = tuple(history) + tuple(target_day)
    X = np.empty((n_phcs, origin_idx.size, len(names)))
    for f, name in enumerate(history):
        X[..., f] = history[name][:, origin_idx]
    for f, name in enumerate(target_day, start=len(history)):
        X[..., f] = target_day[name]

    seasonal_idx = target_idx - seasonal_lag
    baselines = {
        "bl_naive": series[:, origin_idx],
        "bl_seasonal7": np.where(seasonal_idx[None, :] >= 0, series[:, np.maximum(seasonal_idx, 0)], np.nan),
        "bl_mean7": mean7[:, origin_idx],
    }
    return FeatureSet(
        target=target_name, family=target.family, names=names, X=X, y=y,
        origin_idx=origin_idx, horizon=horizon, target_idx=target_idx, start_date=start_date,
        baselines=baselines, target_prevalence=target_prevalence,
        origin_prevalence=prevalence[:, origin_idx], outbreak_threshold=threshold,
        origin_recent_outbreak=recent_outbreak[:, origin_idx],
        target_recent_outbreak=np.where(in_range[None, :], recent_outbreak[:, safe_target], False),
        long_window=long_window,
    )
