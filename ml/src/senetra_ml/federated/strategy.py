"""Aggregation primitives shared by every tier of the federation."""

from __future__ import annotations

import numpy as np


def group_weighted_mean(values: np.ndarray, weights: np.ndarray, groups: np.ndarray,
                        n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean of member vectors per group. Groups without weight get NaN rows."""
    totals = np.bincount(groups, weights=weights, minlength=n_groups)
    sums = np.stack(
        [np.bincount(groups, weights=values[:, f] * weights, minlength=n_groups)
         for f in range(values.shape[1])],
        axis=1,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / totals[:, None]
    means[totals <= 0] = np.nan
    return means, totals


def clip_update_norms(delta: np.ndarray, max_norm: float) -> np.ndarray:
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    return delta * np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))


def gaussian_mechanism(mean_delta: np.ndarray, participants: np.ndarray, clip_norm: float,
                       noise_multiplier: float, rng: np.random.Generator) -> np.ndarray:
    """Adds N(0, (z * C / m)^2) to each group's averaged clipped update (DP-FedAvg style).

    No privacy accountant is run, so no epsilon is claimed.
    """
    std = noise_multiplier * clip_norm / np.maximum(participants, 1.0)
    return mean_delta + rng.normal(size=mean_delta.shape) * std[:, None]
