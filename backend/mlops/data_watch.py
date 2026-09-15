"""New-data detection and the retraining policy.

The signature covers only what training reads: daily metrics, medicine consumption and the PHC list.
Inventory updates change stock, not training data, so they never trigger retraining.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from senetra_ml.data.db import connect_readonly

from mlops.settings import MLOpsSettings

WATCHED_TABLES = (("daily_metrics", "date"), ("medicine_consumption", "date"), ("phcs", None))


@dataclass(frozen=True)
class DataSignature:
    latest_date: str | None
    tables: dict

    def to_json(self) -> str:
        return json.dumps({"latest_date": self.latest_date, "tables": self.tables}, sort_keys=True)

    def as_dict(self) -> dict:
        return json.loads(self.to_json())

    @classmethod
    def from_json(cls, text: str) -> DataSignature:
        data = json.loads(text)
        return cls(latest_date=data.get("latest_date"), tables=data.get("tables", {}))


def read_signature(db_path: str | Path) -> DataSignature:
    conn = connect_readonly(db_path)
    try:
        tables = {}
        for table, date_column in WATCHED_TABLES:
            columns = "COUNT(*), MAX(id)" + (f", MAX({date_column})" if date_column else "")
            row = conn.execute(f"SELECT {columns} FROM {table}").fetchone()
            tables[table] = {"rows": int(row[0]), "max_id": row[1]}
            if date_column:
                tables[table]["max_date"] = row[2][:10] if row[2] else None
    finally:
        conn.close()
    return DataSignature(latest_date=tables["daily_metrics"]["max_date"], tables=tables)


@dataclass(frozen=True)
class RetrainDecision:
    should_retrain: bool
    reason: str
    new_days: int = 0


def new_days_between(current: DataSignature, previous: DataSignature | None) -> int:
    if previous is None or not current.latest_date or not previous.latest_date:
        return 0
    return int((pd.Timestamp(current.latest_date) - pd.Timestamp(previous.latest_date)).days)


def decide_retrain(current: DataSignature, trained: DataSignature | None, settings: MLOpsSettings, *,
                   last_finished_at: datetime | None, canary_active: bool, force: bool,
                   now: datetime) -> RetrainDecision:
    if force:
        return RetrainDecision(True, "manual retraining requested")
    if trained is None:
        return RetrainDecision(True, "no training baseline recorded yet")
    if current.to_json() == trained.to_json():
        return RetrainDecision(False, "no new data since the last training")
    days = new_days_between(current, trained)
    if canary_active:
        return RetrainDecision(False, f"new data ({days} day(s)) waits until the active canary is promoted or "
                                      "rolled back", days)
    if last_finished_at and now - last_finished_at < timedelta(minutes=settings.retrain_cooldown_minutes):
        return RetrainDecision(False, f"cooldown: last retraining finished at {last_finished_at.isoformat()}", days)
    if 0 < days < settings.retrain_min_new_days:
        return RetrainDecision(False, f"{days} new day(s); retraining starts at {settings.retrain_min_new_days}", days)
    if days > 0:
        return RetrainDecision(True, f"{days} new day(s) of data", days)
    return RetrainDecision(True, "existing training data changed (backfill or corrections)", days)
