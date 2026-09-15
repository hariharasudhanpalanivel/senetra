"""Read-only SQL access to the SENETRA operational database.

Data-access contract (mirrors the federated privacy boundary):
- Row-level PHC records are only read through queries scoped to one PHC (`WHERE phc_id = ?`),
  i.e. the reads a PHC client performs on its own data.
- Cross-PHC reads return aggregates only (district outbreak prevalence, reference tables).
- `target_actuals` is the one multi-PHC row read; evaluation and monitoring reduce it
  immediately to additive per-PHC statistics, which is what a client would report.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd

from senetra_ml.config import METRIC_COLUMNS, TargetConfig
from senetra_ml.data.db import connect_readonly


def next_day(date: str | pd.Timestamp) -> str:
    return (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def day(date: str | pd.Timestamp) -> str:
    return pd.Timestamp(date).strftime("%Y-%m-%d")


class SenetraRepository:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if not self.db_path.is_file():
            raise FileNotFoundError(f"SENETRA database not found: {self.db_path}")
        self._conn = connect_readonly(self.db_path)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SenetraRepository:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query(sql, self._conn, params=params)

    def rows(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------------ reference data

    def table_names(self) -> set[str]:
        return {r[0] for r in self.rows("SELECT name FROM sqlite_master WHERE type = 'table'")}

    def date_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        (lo, hi), = self.rows("SELECT MIN(date), MAX(date) FROM daily_metrics")
        if lo is None:
            raise ValueError("daily_metrics is empty")
        return pd.Timestamp(lo[:10]), pd.Timestamp(hi[:10])

    def hierarchy(self) -> pd.DataFrame:
        return self.query(
            """
            SELECT p.id AS phc_id, p.name AS phc_name,
                   d.id AS district_id, d.name AS district_name,
                   s.id AS state_id, s.name AS state_name,
                   c.id AS country_id, c.name AS country_name, c.code AS country_code
            FROM phcs p
            JOIN districts d ON d.id = p.district_id
            JOIN states s ON s.id = d.state_id
            JOIN countries c ON c.id = s.country_id
            ORDER BY c.id, d.id, p.id
            """
        )

    def medicines(self) -> pd.DataFrame:
        return self.query("SELECT id AS medicine_id, name, unit FROM medicines ORDER BY id")

    # ------------------------------------------------------------------ PHC-scoped reads

    def phc_metric_rows(self, phc_id: int, start: str, end: str) -> list[tuple]:
        return self.rows(
            """
            SELECT date, patient_footfall, bed_occupancy, staff_availability, outbreak_flag
            FROM daily_metrics
            WHERE phc_id = ? AND date >= ? AND date < ?
            """,
            (int(phc_id), day(start), next_day(end)),
        )

    def phc_consumption_rows(self, phc_id: int, start: str, end: str) -> list[tuple]:
        return self.rows(
            """
            SELECT date, medicine_id, consumption
            FROM medicine_consumption
            WHERE phc_id = ? AND date >= ? AND date < ?
            """,
            (int(phc_id), day(start), next_day(end)),
        )

    # ------------------------------------------------------------------ aggregate reads

    def district_surveillance(self, start: str, end: str) -> pd.DataFrame:
        """Daily share of reporting PHCs that flag an outbreak, per district (counts only)."""
        frame = self.query(
            """
            SELECT p.district_id, substr(m.date, 1, 10) AS date,
                   AVG(CAST(m.outbreak_flag AS REAL)) AS prevalence,
                   COUNT(*) AS reporting_phcs
            FROM daily_metrics m
            JOIN phcs p ON p.id = m.phc_id
            WHERE m.date >= ? AND m.date < ?
            GROUP BY p.district_id, substr(m.date, 1, 10)
            """,
            (day(start), next_day(end)),
        )
        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def inventory(self) -> pd.DataFrame:
        return self.query(
            """
            SELECT i.phc_id, m.name AS medicine, i.current_stock, i.updated_at
            FROM inventory i JOIN medicines m ON m.id = i.medicine_id
            """
        )

    def active_events(self, as_of: str | pd.Timestamp) -> pd.DataFrame:
        return self.query(
            """
            SELECT district_id, event_type, severity, started_at
            FROM simulation_events
            WHERE active = 1 AND started_at < ?
            """,
            (next_day(as_of),),
        )

    def target_actuals(self, target: TargetConfig, start: str, end: str) -> pd.DataFrame:
        """Realized daily values for one target: columns phc_id, date, actual."""
        if target.source == "daily_metrics":
            if target.column not in METRIC_COLUMNS:
                raise ValueError(f"Unsupported metric column {target.column!r}")
            sql = f"""
                SELECT phc_id, substr(date, 1, 10) AS date, {target.column} AS actual
                FROM daily_metrics WHERE date >= ? AND date < ?
            """
            params: tuple = (day(start), next_day(end))
        else:
            sql = """
                SELECT mc.phc_id, substr(mc.date, 1, 10) AS date, mc.consumption AS actual
                FROM medicine_consumption mc JOIN medicines m ON m.id = mc.medicine_id
                WHERE m.name = ? AND mc.date >= ? AND mc.date < ?
            """
            params = (target.medicine, day(start), next_day(end))
        frame = self.query(sql, params)
        frame["date"] = pd.to_datetime(frame["date"])
        return frame
