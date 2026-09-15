"""Forecast service shared by the CLI and the API.

For a forecast origin it reads each in-scope PHC's recent history from the database, builds
the same features used in training, applies a scenario to the target-day outbreak covariate,
runs the registered model at the requested level and attaches stock-out risk for medicines.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from senetra_ml.config import LEVELS, PipelineConfig
from senetra_ml.data.db import connect_writable
from senetra_ml.data.panel import Panel, load_panel
from senetra_ml.data.repository import SenetraRepository, day
from senetra_ml.features.engineering import FeatureSet, build_features
from senetra_ml.models.bundle import ENTITY_COLUMN
from senetra_ml.risk import days_of_supply, risk_level
from senetra_ml.tracking import LoadedModel, RegistryModelLoader

SCENARIOS = ("observed", "normal", "outbreak", "simulated")
ENTITY_NAME = {"phc": "phc_name", "district": "district_name", "country": "country_name"}


@dataclass(frozen=True)
class Scenario:
    """observed: current outbreak state persists | normal: no outbreak | outbreak: what-if at a
    severity (optionally limited to districts) | simulated: active simulation_events applied."""

    name: str = "observed"
    severity: float | None = None
    district_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.name not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        if self.name == "outbreak" and (self.severity is None or not 0 < self.severity <= 10):
            raise ValueError("the outbreak scenario needs a severity in (0, 10]")

    @property
    def interval_kind(self) -> str:
        return "observed" if self.name == "observed" else "scenario"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["district_ids"] = list(self.district_ids) if self.district_ids else None
        return data


def model_input_frame(fs: FeatureSet, panel: Panel, row_mask: np.ndarray) -> pd.DataFrame:
    ci, ni = np.nonzero(row_mask)
    district_pos = panel.district_index[ci]
    frame = pd.DataFrame({
        "phc_id": panel.phc_ids[ci].astype(np.int64),
        "district_id": panel.district_ids[district_pos].astype(np.int64),
        "country_id": panel.country_ids[panel.district_country_index[district_pos]].astype(np.int64),
        "origin_date": fs.origin_dates[ni].strftime("%Y-%m-%d"),
        "target_date": fs.target_dates[ni].strftime("%Y-%m-%d"),
        "horizon": fs.horizon[ni].astype(np.int64),
    })
    features = fs.X[ci, ni]
    for f, name in enumerate(fs.names):
        frame[name] = features[:, f]
    return frame


def _finite_or_none(value: float | None, digits: int = 2) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


class ForecastService:
    PANEL_TTL_SECONDS = 60.0

    def __init__(self, cfg: PipelineConfig, repo: SenetraRepository | None = None,
                 loader: RegistryModelLoader | None = None, panel_cache: dict | None = None,
                 lock: threading.RLock | None = None):
        """`panel_cache` and `lock` may be shared by several services (e.g. stable and canary models)
        so they reuse the same feature data instead of loading it twice."""
        self.cfg = cfg
        self.repo = repo or SenetraRepository(cfg.data.db_path)
        self.loader = loader or RegistryModelLoader(cfg)
        self._models: dict[str, LoadedModel] = {}
        self._panels: dict[tuple, tuple[float, Panel]] = {} if panel_cache is None else panel_cache
        self._hierarchy: pd.DataFrame | None = None
        self._lock = lock or threading.RLock()

    # ------------------------------------------------------------------ caches

    def reload(self) -> None:
        with self._lock:
            self._models.clear()
            self._panels.clear()
            self._hierarchy = None

    @property
    def hierarchy(self) -> pd.DataFrame:
        with self._lock:
            if self._hierarchy is None:
                self._hierarchy = self.repo.hierarchy()
            return self._hierarchy

    def model(self, target: str) -> LoadedModel:
        self.cfg.target(target)
        with self._lock:
            if target not in self._models:
                self._models[target] = self.loader.load(target)
            return self._models[target]

    def evict_model(self, target: str) -> None:
        """Forget a cached model so the next request loads whatever version the registry alias points to now."""
        with self._lock:
            self._models.pop(target, None)

    def _panel(self, origin: pd.Timestamp, phc_ids: np.ndarray) -> Panel:
        key = (day(origin), hash(tuple(int(p) for p in phc_ids)))
        now = time.monotonic()
        with self._lock:
            cached = self._panels.get(key)
            if cached and now - cached[0] < self.PANEL_TTL_SECONDS:
                return cached[1]
        lookback = self.cfg.forecast.long_window + self.cfg.data.min_history_days + 7
        panel = load_panel(self.repo, origin - pd.Timedelta(days=lookback), origin, phc_ids)
        with self._lock:
            if len(self._panels) >= 16:
                self._panels.pop(next(iter(self._panels)))
            self._panels[key] = (now, panel)
        return panel

    # ------------------------------------------------------------------ scope and scenario

    def scope(self, level: str, entity_id: int | None) -> pd.DataFrame:
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        if level == "world":
            return self.hierarchy
        if entity_id is None:
            raise ValueError(f"entity_id is required for level '{level}'")
        subset = self.hierarchy[self.hierarchy[ENTITY_COLUMN[level]] == int(entity_id)]
        if subset.empty:
            raise LookupError(f"No PHCs found for {level} {entity_id}")
        return subset

    def origin(self, origin: str | pd.Timestamp | None) -> pd.Timestamp:
        lo, hi = self.repo.date_bounds()
        ts = pd.Timestamp(origin).normalize() if origin is not None else hi
        if ts > hi or ts < lo:
            raise ValueError(f"origin {day(ts)} is outside the available data {day(lo)}..{day(hi)}")
        return ts

    def scenario_intensity(self, scenario: Scenario, fs: FeatureSet, panel: Panel,
                           origin: pd.Timestamp) -> np.ndarray:
        """Target-day outbreak intensity per row: 0 normal, 1.0 = the outbreak level seen in history."""
        observed = np.nan_to_num(fs.origin_outbreak(), nan=0.0)
        if scenario.name == "observed":
            return observed
        if scenario.name == "normal":
            return np.zeros_like(observed)
        per_severity = 1.0 / self.cfg.scenarios.severity_reference
        districts = panel.hierarchy["district_id"].to_numpy()
        if scenario.name == "outbreak":
            applies = (np.ones(districts.size, dtype=bool) if not scenario.district_ids
                       else np.isin(districts, scenario.district_ids))
            return np.where(applies[:, None], scenario.severity * per_severity, observed)
        events = self.repo.active_events(origin)
        if events.empty:
            return observed
        severity = events.groupby("district_id")["severity"].max()
        event_intensity = pd.Series(districts).map(severity).fillna(0.0).to_numpy() * per_severity
        return np.maximum(observed, event_intensity[:, None])

    # ------------------------------------------------------------------ forecasting

    def phc_frame(self, target: str, level: str, entity_id: int | None, scenario: Scenario,
                  origin: str | pd.Timestamp | None) -> tuple[pd.DataFrame, pd.Timestamp, pd.DataFrame, Panel, LoadedModel]:
        loaded = self.model(target)
        scope = self.scope(level, entity_id)
        origin_ts = self.origin(origin)
        panel = self._panel(origin_ts, scope["phc_id"].to_numpy())
        fs = build_features(panel, target, self.cfg, origins=np.array([panel.date_index(origin_ts)]))
        fs.X[..., fs.scenario_index] = self.scenario_intensity(scenario, fs, panel, origin_ts)
        frame = model_input_frame(fs, panel, np.ones(fs.y.shape, dtype=bool))
        return frame, origin_ts, scope, panel, loaded

    def forecast(self, target: str, level: str = "district", entity_id: int | None = None,
                 scenario: Scenario | None = None, origin: str | pd.Timestamp | None = None) -> dict:
        scenario = scenario or Scenario()
        target_cfg = self.cfg.target(target)
        frame, origin_ts, scope, panel, loaded = self.phc_frame(target, level, entity_id, scenario, origin)
        out = loaded.bundle.forecast(frame, level=level, interval_kind=scenario.interval_kind)
        out = out.sort_values(["entity_id", "horizon"])

        records = []
        for row in out.itertuples(index=False):
            records.append({
                "date": row.target_date, "horizon": int(row.horizon), "yhat": round(float(row.yhat), 2),
                "lower_80": round(float(row.lo80), 2), "upper_80": round(float(row.hi80), 2),
                "lower_95": round(float(row.lo95), 2), "upper_95": round(float(row.hi95), 2),
                "confidence": round(float(row.confidence), 3), "imputed_share": round(float(row.imputed_share), 3),
            })
        name_column = ENTITY_NAME.get(level)
        response = {
            "target": target,
            "unit": target_cfg.unit or panel.medicine_units.get(target_cfg.medicine or "", None),
            "level": level,
            "entity_id": int(entity_id) if level != "world" else None,
            "entity_name": str(scope[name_column].iloc[0]) if name_column else "World",
            "phcs_in_scope": int(len(scope)),
            "origin_date": day(origin_ts),
            "scenario": scenario.to_dict(),
            "model": {"name": loaded.name, "version": loaded.version, "alias": loaded.alias,
                      "parameter_level": loaded.bundle.selected_level},
            "forecast": records,
            "horizon_total": round(float(out["yhat"].sum()), 2) if target_cfg.aggregation == "sum" else None,
        }
        if target_cfg.is_medicine:
            response["stock"] = self._stock_risk(target_cfg.medicine, scope["phc_id"], out)
        return response

    def forecast_all(self, level: str = "district", entity_id: int | None = None,
                     scenario: Scenario | None = None, origin: str | pd.Timestamp | None = None) -> dict:
        results, errors = {}, {}
        for target in self.cfg.targets:
            try:
                results[target] = self.forecast(target, level, entity_id, scenario, origin)
            except Exception as exc:  # one missing model must not hide the other targets
                errors[target] = str(exc)
        return {"forecasts": results, "errors": errors}

    def _stock_risk(self, medicine: str, phc_ids: pd.Series, out: pd.DataFrame) -> dict:
        inventory = self.repo.inventory()
        stock = inventory.loc[(inventory["medicine"] == medicine) & inventory["phc_id"].isin(phc_ids),
                              "current_stock"].sum()
        central = days_of_supply(float(stock), out["yhat"].to_numpy())
        pessimistic = days_of_supply(float(stock), out["hi80"].to_numpy())
        return {
            "current_stock": round(float(stock), 2),
            "days_of_supply": _finite_or_none(central),
            "days_of_supply_pessimistic": _finite_or_none(pessimistic),
            "risk_level": risk_level(central, self.cfg.risk),
            "risk_level_pessimistic": risk_level(pessimistic, self.cfg.risk),
        }

    # ------------------------------------------------------------------ batch writes

    def write_predictions(self, target: str, scenario: Scenario,
                          origin: str | pd.Timestamp | None = None) -> int:
        """Store PHC-level medicine forecasts in the `predictions` table (idempotent per model version)."""
        target_cfg = self.cfg.target(target)
        if not target_cfg.is_medicine:
            raise ValueError(f"{target} is not a medicine target; the predictions table stores medicines only")
        frame, _, _, _, loaded = self.phc_frame(target, "world", None, scenario, origin)
        out = loaded.bundle.forecast(frame, level="phc", interval_kind=scenario.interval_kind)
        (medicine_id,), = self.repo.rows("SELECT id FROM medicines WHERE name = ?", (target_cfg.medicine,))
        columns = {row[1] for row in self.repo.rows("PRAGMA table_info(predictions)")}
        with_bounds = {"lower_bound", "upper_bound"} <= columns
        label = loaded.label
        rows = [
            (int(r.entity_id), int(medicine_id), r.target_date, float(r.yhat), float(r.confidence), label)
            + ((float(r.lo80), float(r.hi80)) if with_bounds else ())
            for r in out.itertuples(index=False)
        ]
        insert = ("INSERT INTO predictions (phc_id, medicine_id, prediction_date, predicted_demand, confidence,"
                  " model_version" + (", lower_bound, upper_bound) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                                      if with_bounds else ") VALUES (?, ?, ?, ?, ?, ?)"))
        conn = connect_writable(self.cfg.data.db_path)
        try:
            with conn:
                conn.execute(
                    "DELETE FROM predictions WHERE medicine_id = ? AND model_version = ?"
                    " AND prediction_date BETWEEN ? AND ?",
                    (int(medicine_id), label, out["target_date"].min(), out["target_date"].max()),
                )
                conn.executemany(insert, rows)
        finally:
            conn.close()
        return len(rows)
