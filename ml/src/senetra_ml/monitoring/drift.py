"""Regime-matched distribution profiles and population stability index (PSI).

Outbreaks shift demand by design, and rolling-window features keep that shift for weeks after an
outbreak ends. Rows are therefore compared only with reference rows of the same regime:
- outbreak: the district is in an outbreak on the origin or target day;
- recovery: an outbreak day lies inside the long history window;
- normal: neither.
Reference rows with incomplete history windows (warm-up) are excluded. Bins are fixed around the
federated mean/std, so each client can count its own rows and the aggregator only sums counts.
"""

from __future__ import annotations

import numpy as np

from senetra_ml.features.engineering import FeatureSet
from senetra_ml.features.transform import FeatureTransform

Z_EDGES = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
REGIME_FEATURES = {"scenario_outbreak", "dprev_origin", "dprev_mean7", "flag_share7"}
SKIP_FEATURES = {"horizon", "dow_sin", "dow_cos"}
TARGET_KEY = "__target__"
REGIMES = ("normal", "recovery", "outbreak")


WINDOW_TOKENS = ("mean", "trend", "disp", "share", "dprev")


def is_window_feature(name: str) -> bool:
    """Rolling-window features: during recovery their values depend on days since the outbreak ended."""
    return any(token in name for token in WINDOW_TOKENS)


def bin_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    return np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)


def psi(expected: np.ndarray, actual_counts: np.ndarray, eps: float = 1e-4) -> float:
    total = actual_counts.sum()
    if total == 0:
        return float("nan")
    expected = np.clip(np.asarray(expected, dtype=float), eps, None)
    actual = np.clip(actual_counts / total, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def row_regimes(fs: FeatureSet) -> tuple[np.ndarray, np.ndarray]:
    """(C, N) regime index (into REGIMES) of each row's features and of its target day."""
    threshold = fs.outbreak_threshold
    origin_outbreak = np.nan_to_num(fs.origin_prevalence) > threshold
    target_outbreak = np.nan_to_num(fs.target_prevalence) > threshold
    features = np.where(origin_outbreak | target_outbreak, 2, np.where(fs.origin_recent_outbreak, 1, 0))
    target = np.where(target_outbreak, 2, np.where(fs.target_recent_outbreak, 1, 0))
    return features, target


def regime_samples(fs: FeatureSet, transform: FeatureTransform, feature_mask: np.ndarray,
                   target_mask: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    feature_regime, target_regime = row_regimes(fs)
    samples: dict[str, dict[str, np.ndarray]] = {}
    for code, regime in enumerate(REGIMES):
        rows = feature_mask & (feature_regime == code)
        values = {}
        for f, name in enumerate(transform.names):
            if name in SKIP_FEATURES:
                continue
            column = fs.X[..., f][rows]
            values[name] = transform.intensity(column) if f == transform.scenario_index else column
        values[TARGET_KEY] = fs.y[target_mask & (target_regime == code)]
        samples[regime] = values
    return samples


def build_reference_profile(fs: FeatureSet, mask: np.ndarray, transform: FeatureTransform) -> dict:
    mask = mask & (fs.origin_idx >= fs.long_window - 1)[None, :]
    y = fs.y[mask]
    y_scale = float(np.std(y)) or 1.0
    edges = {name: (float(transform.mean[f]) + float(transform.scale[f]) * Z_EDGES).tolist()
             for f, name in enumerate(transform.names) if name not in SKIP_FEATURES}
    edges[TARGET_KEY] = (float(np.mean(y)) + y_scale * Z_EDGES).tolist()

    profile = {"edges": edges, "regime_features": sorted(REGIME_FEATURES), "regimes": {}}
    for regime, values in regime_samples(fs, transform, mask, mask).items():
        proportions = {}
        for name, column in values.items():
            counts = bin_counts(column, np.asarray(edges[name]))
            proportions[name] = (counts / max(counts.sum(), 1)).tolist()
        profile["regimes"][regime] = {"rows": int(values[TARGET_KEY].size), "proportions": proportions}
    total = sum(r["rows"] for r in profile["regimes"].values())
    profile["outbreak_share"] = profile["regimes"]["outbreak"]["rows"] / total if total else 0.0
    return profile


def compare_to_profile(profile: dict, samples: dict[str, dict[str, np.ndarray]],
                       min_rows: int = 50) -> dict[str, dict[str, dict]]:
    """PSI per regime and feature; regimes lacking enough reference or recent rows are skipped."""
    out: dict[str, dict[str, dict]] = {}
    for regime, values in samples.items():
        reference = profile["regimes"].get(regime, {})
        if reference.get("rows", 0) < min_rows or values[TARGET_KEY].size < min_rows:
            continue
        out[regime] = {}
        for name, column in values.items():
            if name not in reference["proportions"]:
                continue
            counts = bin_counts(column, np.asarray(profile["edges"][name]))
            out[regime][name] = {
                "psi": psi(np.asarray(reference["proportions"][name]), counts),
                "n": int(counts.sum()),
                "regime_feature": name in REGIME_FEATURES,
                # Recovery mixes early and late post-outbreak days, so window features drift by construction.
                "phase_dependent": regime == "recovery" and name != TARGET_KEY and is_window_feature(name),
            }
    return out
