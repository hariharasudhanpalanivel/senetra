"""Rolling-origin backtesting and model fitting for one target.

Fold k has cutoff c_k: models see only rows whose target day is on or before c_k, and are
tested on origins c_k .. c_k + test_origin_days - 1 (every horizon). Cutoffs step back from the
end of the data so folds cover the latest period, recoveries, active outbreaks and, when the
history allows, a cold start where training contains no outbreak at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from senetra_ml.config import LEVELS, PipelineConfig
from senetra_ml.data.panel import Panel
from senetra_ml.features.engineering import BASELINES, FeatureSet
from senetra_ml.features.transform import FeatureTransform, fit_transform
from senetra_ml.federated.glm import GLMProblem, inverse_link
from senetra_ml.federated.trainer import FederatedModel, HierarchicalFederatedTrainer, initial_params
from senetra_ml.models.centralized import fit_xgboost, predict_xgboost

SCENARIO_KINDS = ("oracle", "observed")


@dataclass(frozen=True)
class Fold:
    number: int
    cutoff_idx: int
    test_start_idx: int
    test_end_idx: int


@dataclass
class FittedModels:
    cutoff_idx: int
    transform: FeatureTransform
    federated: FederatedModel
    local_only: np.ndarray | None
    booster: xgb.Booster | None
    train_rows: int
    outbreak_district_days: int

    @property
    def cold_start(self) -> bool:
        return self.outbreak_district_days == 0


def make_folds(n_dates: int, first_origin: int, cfg: PipelineConfig) -> list[Fold]:
    horizon, test_days = cfg.forecast.horizon_days, cfg.backtest.test_origin_days
    cutoffs = []
    for k in range(cfg.backtest.n_folds):
        cutoff = n_dates - horizon - test_days - k * cfg.backtest.step_days
        if cutoff - first_origin < max(7, horizon):
            break
        cutoffs.append(cutoff)
    return [Fold(i, c, c, c + test_days - 1) for i, c in enumerate(sorted(cutoffs))]


def fit_models(fs: FeatureSet, panel: Panel, cutoff_idx: int, cfg: PipelineConfig,
               history_ok: np.ndarray, include_benchmarks: bool) -> FittedModels:
    mask = fs.train_mask(cutoff_idx, history_ok)
    if not mask.any():
        raise ValueError(f"No training rows for {fs.target} at cutoff {panel.dates[cutoff_idx].date()}")
    threshold = cfg.data.outbreak_prevalence_threshold
    transform = fit_transform(fs.X, mask, fs.names, fs.scenario_index, cfg.scenarios.max_intensity)
    Z, _ = transform.apply(fs.X)
    problem = GLMProblem.build(Z, fs.y, mask, fs.family, cfg.federated.l2)
    init = initial_params(problem)
    trainer = HierarchicalFederatedTrainer(cfg.federated)
    federated = trainer.fit(
        problem, init, phc_ids=panel.phc_ids, district_ids=panel.district_ids,
        district_index=panel.district_index, country_ids=panel.country_ids,
        district_country_index=panel.district_country_index,
    )
    local_only = trainer.fit_local_only(problem, init) if include_benchmarks else None
    booster = None
    if include_benchmarks and cfg.benchmarks.xgboost.enabled:
        booster = fit_xgboost(fs, transform, mask, cutoff_idx, cfg.benchmarks.xgboost, cfg.federated.seed)
    outbreak_days = int((np.nan_to_num(panel.district_prevalence[:, : cutoff_idx + 1]) > threshold).sum())
    return FittedModels(cutoff_idx, transform, federated, local_only, booster, int(mask.sum()), outbreak_days)


def predict_variants(fs: FeatureSet, panel: Panel, fitted: FittedModels, row_mask: np.ndarray,
                     cfg: PipelineConfig) -> pd.DataFrame:
    """PHC-level predictions of every model variant under oracle and observed scenarios."""
    target_cfg = cfg.target(fs.target)
    ci, ni = np.nonzero(row_mask)
    X_oracle = fs.X[ci, ni]
    X_observed = X_oracle.copy()
    X_observed[:, fs.scenario_index] = np.nan_to_num(fs.origin_outbreak()[ci, ni])
    designs = {"oracle": fitted.transform.apply(X_oracle)[0], "observed": fitted.transform.apply(X_observed)[0]}
    raw = {"oracle": X_oracle, "observed": X_observed}

    threshold = cfg.data.outbreak_prevalence_threshold
    district_pos = panel.district_index[ci]
    frame = pd.DataFrame({
        "phc_id": panel.phc_ids[ci],
        "district_id": panel.district_ids[district_pos],
        "country_id": panel.country_ids[panel.district_country_index[district_pos]],
        "origin_date": fs.origin_dates[ni],
        "horizon": fs.horizon[ni],
        "target_date": fs.target_dates[ni],
        "outbreak": fs.target_prevalence[ci, ni] > threshold,
        "origin_outbreak": fs.origin_prevalence[ci, ni] > threshold,
    })
    frame["transition"] = frame["outbreak"] != frame["origin_outbreak"]

    def finish(pred: np.ndarray) -> np.ndarray:
        return np.clip(pred, *target_cfg.bounds) if target_cfg.bounds else pred

    for level in LEVELS:
        params = fitted.federated.client_params(level, panel.district_index, panel.district_country_index)[ci]
        for kind in SCENARIO_KINDS:
            frame[f"fl_{level}__{kind}"] = finish(
                inverse_link(np.einsum("mf,mf->m", designs[kind], params), fs.family))
    if fitted.local_only is not None:
        params = fitted.local_only[ci]
        for kind in SCENARIO_KINDS:
            frame[f"local__{kind}"] = finish(inverse_link(np.einsum("mf,mf->m", designs[kind], params), fs.family))
    if fitted.booster is not None:
        for kind in SCENARIO_KINDS:
            frame[f"xgb__{kind}"] = finish(predict_xgboost(fitted.booster, raw[kind], fitted.transform, fs.names))
    for name in BASELINES:
        frame[name] = fs.baselines[name][ci, ni]
    return frame


def run_backtest(fs: FeatureSet, panel: Panel, cfg: PipelineConfig,
                 progress=None) -> tuple[pd.DataFrame, list[dict]]:
    history_ok = fs.history_ok()
    folds = make_folds(panel.n_dates, int(fs.origin_idx.min()), cfg)
    if not folds:
        raise ValueError("Not enough history for a single backtest fold")
    known = np.isfinite(fs.y) & np.isfinite(fs.X[..., fs.scenario_index])
    frames, summaries = [], []
    for fold in folds:
        fitted = fit_models(fs, panel, fold.cutoff_idx, cfg, history_ok, include_benchmarks=True)
        in_test = (fs.origin_idx >= fold.test_start_idx) & (fs.origin_idx <= fold.test_end_idx)
        test_mask = history_ok & known & in_test[None, :]
        frame = predict_variants(fs, panel, fitted, test_mask, cfg)
        frame["actual"] = fs.y[test_mask]
        frame["fold"] = fold.number
        frame["cold_start"] = fitted.cold_start
        frames.append(frame)
        booster = fitted.booster
        summary = {
            "fold": fold.number,
            "cutoff_date": panel.dates[fold.cutoff_idx].strftime("%Y-%m-%d"),
            "test_origins": f"{panel.dates[fold.test_start_idx].date()}..{panel.dates[fold.test_end_idx].date()}",
            "train_rows": fitted.train_rows,
            "test_rows": int(test_mask.sum()),
            "cold_start": fitted.cold_start,
            "outbreak_district_days_in_train": fitted.outbreak_district_days,
            "federated_history": fitted.federated.history,
            "xgb_best_iteration": getattr(booster, "best_iteration", None) if booster is not None else None,
        }
        summaries.append(summary)
        if progress:
            progress(summary)
    return pd.concat(frames, ignore_index=True), summaries
