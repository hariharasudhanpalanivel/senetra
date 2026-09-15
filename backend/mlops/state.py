"""Persistent MLOps state shared by every backend process and retraining job.

A small SQLite file (default backend/instance/mlops.db) holds jobs, data watermarks, the prediction
log used to judge canaries, deployment events and the scheduler lease. The operational senetra.db is
never written here.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS ml_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    trigger TEXT NOT NULL,
    status TEXT NOT NULL,
    targets TEXT,
    force INTEGER NOT NULL DEFAULT 0,
    pid INTEGER,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    heartbeat_at TEXT,
    data_signature TEXT,
    result TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_ml_jobs_kind_status ON ml_jobs (kind, status);

CREATE TABLE IF NOT EXISTS ml_data_watermarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature TEXT NOT NULL,
    latest_date TEXT,
    job_id INTEGER,
    baseline INTEGER NOT NULL DEFAULT 0,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ml_prediction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    target TEXT NOT NULL,
    variant TEXT NOT NULL,
    model_version TEXT,
    level TEXT,
    entity_id INTEGER,
    scenario TEXT,
    status TEXT NOT NULL,
    latency_ms REAL,
    forced INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ml_prediction_log ON ml_prediction_log (target, variant, model_version, created_at);

CREATE TABLE IF NOT EXISTS ml_deployment_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    target TEXT NOT NULL,
    model_version TEXT,
    action TEXT NOT NULL,
    percent INTEGER,
    details TEXT
);

CREATE TABLE IF NOT EXISTS ml_leases (
    name TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    expires_at REAL NOT NULL
);
"""

JOB_FIELDS = {"status", "pid", "started_at", "finished_at", "heartbeat_at", "result", "error"}
ACTIVE_STATUSES = ("queued", "running")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _loads(value):
    return json.loads(value) if value else None


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ jobs

    @staticmethod
    def _job(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        job = dict(row)
        job["targets"] = _loads(job["targets"])
        job["result"] = _loads(job["result"])
        job["force"] = bool(job["force"])
        job.pop("data_signature", None)
        return job

    def create_job(self, kind: str, trigger: str, targets: list[str] | None, force: bool,
                   data_signature: str | None) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO ml_jobs (kind, trigger, status, targets, force, created_at, data_signature)"
                " VALUES (?, ?, 'queued', ?, ?, ?, ?)",
                (kind, trigger, json.dumps(list(targets)) if targets else None, int(force), utcnow(), data_signature),
            )
            return int(cursor.lastrowid)

    def update_job(self, job_id: int, **values) -> None:
        unknown = set(values) - JOB_FIELDS
        if unknown:
            raise ValueError(f"Unknown job fields: {sorted(unknown)}")
        assignments = ", ".join(f"{name} = ?" for name in values)
        with self._connect() as conn:
            conn.execute(f"UPDATE ml_jobs SET {assignments} WHERE id = ?", (*values.values(), job_id))

    def get_job(self, job_id: int) -> dict | None:
        with self._connect() as conn:
            return self._job(conn.execute("SELECT * FROM ml_jobs WHERE id = ?", (job_id,)).fetchone())

    def list_jobs(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM ml_jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._job(row) for row in rows]

    def active_job(self, kind: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ml_jobs WHERE kind = ? AND status IN (?, ?) ORDER BY id DESC LIMIT 1",
                (kind, *ACTIVE_STATUSES),
            ).fetchone()
        return self._job(row)

    def last_finished_job(self, kind: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ml_jobs WHERE kind = ? AND finished_at IS NOT NULL ORDER BY id DESC LIMIT 1", (kind,)
            ).fetchone()
        return self._job(row)

    # ------------------------------------------------------------------ data watermarks

    def record_signature(self, signature: str, latest_date: str | None, job_id: int | None,
                         baseline: bool = False) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ml_data_watermarks (signature, latest_date, job_id, baseline, recorded_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (signature, latest_date, job_id, int(baseline), utcnow()),
            )

    def last_signature(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM ml_data_watermarks ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ prediction log

    def log_prediction(self, *, target: str, variant: str, model_version: str | None, level: str | None,
                       entity_id: int | None, scenario: str | None, status: str, latency_ms: float | None,
                       forced: bool = False) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ml_prediction_log (created_at, target, variant, model_version, level, entity_id,"
                " scenario, status, latency_ms, forced) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (utcnow(), target, variant, model_version, level, entity_id, scenario, status, latency_ms, int(forced)),
            )

    def prediction_stats(self, target: str, variant: str, model_version: str | None, since: str | None) -> dict:
        """Requests, errors and p95 latency of one model version since `since` (forced and client errors excluded)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, latency_ms FROM ml_prediction_log WHERE target = ? AND variant = ?"
                " AND model_version = ? AND created_at >= ? AND forced = 0 AND status != 'client_error'"
                " ORDER BY id DESC LIMIT 5000",
                (target, variant, model_version, since or ""),
            ).fetchall()
        requests = len(rows)
        errors = sum(1 for row in rows if row["status"] != "ok")
        latencies = [row["latency_ms"] for row in rows if row["status"] == "ok" and row["latency_ms"] is not None]
        return {
            "requests": requests,
            "errors": errors,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
        }

    # ------------------------------------------------------------------ deployment events

    def add_event(self, target: str, model_version: str | None, action: str, percent: int | None = None,
                  details: dict | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ml_deployment_events (created_at, target, model_version, action, percent, details)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (utcnow(), target, model_version, action, percent, json.dumps(details, default=str) if details else None),
            )

    def list_events(self, target: str | None = None, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            if target:
                rows = conn.execute("SELECT * FROM ml_deployment_events WHERE target = ? ORDER BY id DESC LIMIT ?",
                                    (target, limit)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM ml_deployment_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(row), "details": _loads(row["details"])} for row in rows]

    # ------------------------------------------------------------------ scheduler lease

    def try_acquire_lease(self, name: str, owner: str, ttl_seconds: float) -> bool:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ml_leases (name, owner, expires_at) VALUES (?, ?, ?)"
                " ON CONFLICT(name) DO UPDATE SET owner = excluded.owner, expires_at = excluded.expires_at"
                " WHERE ml_leases.expires_at < ? OR ml_leases.owner = excluded.owner",
                (name, owner, now + ttl_seconds, now),
            )
            row = conn.execute("SELECT owner FROM ml_leases WHERE name = ?", (name,)).fetchone()
        return row is not None and row["owner"] == owner

    def lease_owner(self, name: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT owner, expires_at FROM ml_leases WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def release_lease(self, name: str, owner: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM ml_leases WHERE name = ? AND owner = ?", (name, owner))
