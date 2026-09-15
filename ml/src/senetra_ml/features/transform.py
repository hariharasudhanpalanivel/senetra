"""Fitted feature transform: scenario intensity capping and federated standardization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureTransform:
    names: list[str]
    scenario_index: int
    max_intensity: float
    mean: np.ndarray  # (F,)
    scale: np.ndarray  # (F,)

    def intensity(self, values: np.ndarray) -> np.ndarray:
        """Outbreak intensity: 0 normal, 1 the outbreak level seen in history, capped at max_intensity."""
        return np.clip(values, 0.0, self.max_intensity)

    def scale_scenario(self, X: np.ndarray) -> np.ndarray:
        out = np.array(X, dtype=float, copy=True)
        out[..., self.scenario_index] = self.intensity(out[..., self.scenario_index])
        return out

    def apply(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Standardized design with a leading intercept column, plus a mask of imputed rows."""
        out = np.empty(X.shape[:-1] + (X.shape[-1] + 1,))
        out[..., 0] = 1.0
        body = out[..., 1:]
        body[...] = X
        body[..., self.scenario_index] = self.intensity(body[..., self.scenario_index])
        body -= self.mean
        body /= self.scale
        finite = np.isfinite(body)
        imputed = ~finite.all(axis=-1)
        # Missing inputs fall back to the training mean (0 after standardization).
        body[~finite] = 0.0
        return out, imputed

    def to_dict(self) -> dict:
        return {"names": list(self.names), "scenario_index": self.scenario_index,
                "max_intensity": self.max_intensity, "mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> FeatureTransform:
        return cls(names=list(data["names"]), scenario_index=int(data["scenario_index"]),
                   max_intensity=float(data["max_intensity"]),
                   mean=np.asarray(data["mean"], dtype=float), scale=np.asarray(data["scale"], dtype=float))


def client_moments(X: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-client sufficient statistics (n, sum x, sum x^2) — the only thing clients share."""
    n_clients, _, n_features = X.shape
    s1 = np.zeros((n_clients, n_features))
    s2 = np.zeros((n_clients, n_features))
    for f in range(n_features):
        values = np.where(mask, X[..., f], 0.0)
        s1[:, f] = values.sum(axis=1)
        s2[:, f] = (values * values).sum(axis=1)
    return mask.sum(axis=1).astype(float), s1, s2


def fit_transform(X: np.ndarray, mask: np.ndarray, names: tuple[str, ...], scenario_index: int,
                  max_intensity: float) -> FeatureTransform:
    n, s1, s2 = client_moments(X, mask)
    total = n.sum()
    if total == 0:
        raise ValueError("No training rows available to fit the feature transform")
    mean = s1.sum(axis=0) / total
    std = np.sqrt(np.clip(s2.sum(axis=0) / total - mean * mean, 0.0, None))
    return FeatureTransform(names=list(names), scenario_index=scenario_index, max_intensity=max_intensity,
                            mean=mean, scale=np.where(std > 1e-9, std, 1.0))
