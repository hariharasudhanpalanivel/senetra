"""Centralized XGBoost benchmark.

It pools rows from every PHC, which the federated privacy contract does not allow in
production. It is trained only during backtesting to measure the accuracy cost of federation.
"""

from __future__ import annotations

import os

import numpy as np
import xgboost as xgb

from senetra_ml.config import XGBoostConfig
from senetra_ml.features.engineering import FeatureSet
from senetra_ml.features.transform import FeatureTransform


def fit_xgboost(fs: FeatureSet, transform: FeatureTransform, train_mask: np.ndarray, cutoff_idx: int,
                cfg: XGBoostConfig, seed: int) -> xgb.Booster:
    rows = transform.scale_scenario(fs.X[train_mask])
    y = fs.y[train_mask]
    client = np.broadcast_to(np.arange(fs.y.shape[0])[:, None], fs.y.shape)[train_mask]
    params = {
        "objective": "count:poisson" if fs.family == "poisson" else "reg:squarederror",
        "eval_metric": "mae",
        "seed": seed,
        "nthread": os.cpu_count() or 1,
        **cfg.params,
    }
    names = list(fs.names)
    # Early stopping on a held-out 10% of PHCs. Every row is already before the cutoff, so this does
    # not leak the test period; a time-ordered slice would hide a recent outbreak from training.
    holdout = np.random.default_rng(seed).random(fs.y.shape[0]) < 0.1
    valid = holdout[client]
    if valid.sum() >= 100 and (~valid).sum() >= 100:
        dtrain = xgb.DMatrix(rows[~valid], label=y[~valid], feature_names=names)
        dvalid = xgb.DMatrix(rows[valid], label=y[valid], feature_names=names)
        return xgb.train(params, dtrain, num_boost_round=cfg.num_boost_round, evals=[(dvalid, "valid")],
                         early_stopping_rounds=cfg.early_stopping_rounds, verbose_eval=False)
    dtrain = xgb.DMatrix(rows, label=y, feature_names=names)
    return xgb.train(params, dtrain, num_boost_round=cfg.num_boost_round, verbose_eval=False)


def predict_xgboost(booster: xgb.Booster, X_rows: np.ndarray, transform: FeatureTransform,
                    names: tuple[str, ...]) -> np.ndarray:
    matrix = xgb.DMatrix(transform.scale_scenario(X_rows), feature_names=list(names))
    try:
        best = booster.best_iteration
    except AttributeError:
        best = None
    if best is None:
        return booster.predict(matrix)
    return booster.predict(matrix, iteration_range=(0, int(best) + 1))
