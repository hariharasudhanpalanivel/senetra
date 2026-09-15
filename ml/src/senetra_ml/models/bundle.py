"""Deployable forecast bundle: fitted transform, hierarchical parameters and calibration.

A PHC forecast uses the parameters of the selected federation level, falling back to coarser
levels for entities that were not part of training (e.g. a newly onboarded PHC gets its
district model). District, country and world forecasts are bottom-up sums (or means for
percentage targets) of PHC forecasts, so every level is coherent with the level below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from senetra_ml.config import LEVELS
from senetra_ml.evaluation.calibration import QUANTILES, interval_bounds
from senetra_ml.features.transform import FeatureTransform
from senetra_ml.federated.glm import inverse_link

ID_COLUMNS = ("phc_id", "district_id", "country_id")
KEY_COLUMNS = ("origin_date", "target_date", "horizon")
ENTITY_COLUMN = {"phc": "phc_id", "district": "district_id", "country": "country_id", "world": None}
OUTPUT_COLUMNS = ["level", "entity_id", "origin_date", "target_date", "horizon", "yhat",
                  "lo80", "hi80", "lo95", "hi95", "confidence", "n_phcs", "imputed_share", "level_used"]
PARAM_ARRAYS = ("world", "country_ids", "country", "district_ids", "district", "phc_ids", "phc")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    raise TypeError(f"Not JSON serializable: {type(value)}")


@dataclass
class ForecastBundle:
    target: str
    target_config: dict
    transform: FeatureTransform
    selected_level: str
    world: np.ndarray
    country_ids: np.ndarray
    country: np.ndarray
    district_ids: np.ndarray
    district: np.ndarray
    phc_ids: np.ndarray
    phc: np.ndarray
    conformal: dict
    metadata: dict = field(default_factory=dict)
    reference_profile: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Lookups use binary search, so every id table is stored sorted.
        for ids_name, table_name in (("country_ids", "country"), ("district_ids", "district"), ("phc_ids", "phc")):
            ids = np.asarray(getattr(self, ids_name), dtype=np.int64)
            order = np.argsort(ids, kind="stable")
            setattr(self, ids_name, ids[order])
            setattr(self, table_name, np.asarray(getattr(self, table_name), dtype=float)[order])
        self.world = np.asarray(self.world, dtype=float)

    @property
    def family(self) -> str:
        return self.target_config["family"]

    @property
    def aggregation(self) -> str:
        return self.target_config["aggregation"]

    @property
    def bounds(self) -> tuple[float, float] | None:
        bounds = self.target_config.get("bounds")
        return (float(bounds[0]), float(bounds[1])) if bounds else None

    def params_for(self, phc_ids: np.ndarray, district_ids: np.ndarray,
                   country_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(phc_ids)
        params = np.repeat(self.world[None], n, axis=0)
        used = np.full(n, "world", dtype=object)
        allowed = LEVELS[LEVELS.index(self.selected_level):]
        for level, ids, lookup, table in (("country", country_ids, self.country_ids, self.country),
                                          ("district", district_ids, self.district_ids, self.district),
                                          ("phc", phc_ids, self.phc_ids, self.phc)):
            if level not in allowed or len(lookup) == 0:
                continue
            ids = np.asarray(ids, dtype=np.int64)
            pos = np.clip(np.searchsorted(lookup, ids), 0, len(lookup) - 1)
            hit = lookup[pos] == ids
            params[hit] = table[pos[hit]]
            used[hit] = level
        return params, used

    def predict_rows(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        missing = [c for c in (*ID_COLUMNS, *self.transform.names) if c not in frame.columns]
        if missing:
            raise ValueError(f"Model input is missing columns: {missing}")
        Z, imputed = self.transform.apply(frame[self.transform.names].to_numpy(dtype=float))
        params, used = self.params_for(*(frame[c].to_numpy(dtype=np.int64) for c in ID_COLUMNS))
        yhat = inverse_link(np.einsum("rf,rf->r", Z, params), self.family)
        if self.bounds:
            yhat = np.clip(yhat, *self.bounds)
        return yhat, used, imputed

    def forecast(self, frame: pd.DataFrame, level: str = "phc", interval_kind: str = "scenario") -> pd.DataFrame:
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        if interval_kind not in self.conformal:
            raise ValueError(f"interval_kind must be one of {sorted(self.conformal)}")
        yhat, used, imputed = self.predict_rows(frame)
        rows = frame[list(ID_COLUMNS + KEY_COLUMNS)].copy()
        rows["yhat"], rows["imputed"], rows["level_used"] = yhat, imputed.astype(float), used

        if level == "phc":
            out = rows.assign(entity_id=rows["phc_id"], n_phcs=1, imputed_share=rows["imputed"])
        else:
            entity = ENTITY_COLUMN[level]
            group = ([entity] if entity else []) + list(KEY_COLUMNS)
            out = rows.groupby(group, sort=True).agg(
                yhat=("yhat", self.aggregation), n_phcs=("yhat", "size"), imputed_share=("imputed", "mean")
            ).reset_index()
            out["entity_id"] = out[entity] if entity else 0
            out["level_used"] = self.selected_level

        pred = out["yhat"].to_numpy(dtype=float)
        table = self.conformal[interval_kind].get(level) or self.conformal[interval_kind].get("phc", {})
        bounds = interval_bounds(pred, out["horizon"].to_numpy(dtype=int), table)
        low, high = self.bounds if self.bounds else (0.0, np.inf)
        for key in QUANTILES:
            out[key] = np.clip(bounds[key], low, high)
        width = out["hi80"].to_numpy() - out["lo80"].to_numpy()
        out["confidence"] = np.clip(1.0 - width / (2.0 * np.maximum(np.abs(pred), 1.0)), 0.0, 1.0)
        out["level"] = level
        return out[OUTPUT_COLUMNS].reset_index(drop=True)

    # ------------------------------------------------------------------ persistence

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "params.npz", **{name: getattr(self, name) for name in PARAM_ARRAYS})
        payload = {
            "target": self.target, "target_config": self.target_config,
            "transform": self.transform.to_dict(), "selected_level": self.selected_level,
            "conformal": self.conformal, "metadata": self.metadata,
            "reference_profile": self.reference_profile,
        }
        (directory / "bundle.json").write_text(json.dumps(payload, indent=2, default=_json_default),
                                               encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> ForecastBundle:
        directory = Path(directory)
        payload = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
        with np.load(directory / "params.npz") as arrays:
            params = {name: arrays[name] for name in PARAM_ARRAYS}
        return cls(
            target=payload["target"], target_config=payload["target_config"],
            transform=FeatureTransform.from_dict(payload["transform"]),
            selected_level=payload["selected_level"], conformal=payload["conformal"],
            metadata=payload.get("metadata", {}), reference_profile=payload.get("reference_profile", {}),
            **params,
        )
