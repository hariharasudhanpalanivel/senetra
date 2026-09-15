"""Backend MLOps fixtures: a synthetic database with late-arriving data and isolated MLflow/state stores."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

# Importing app.py builds a module-level app: keep it away from real state, schedulers and MLflow stores.
os.environ["ML_SCHEDULER_ENABLED"] = "false"
os.environ["ML_STATE_DB"] = str(Path(tempfile.mkdtemp(prefix="senetra-mlops-")) / "import.db")
os.environ.pop("MLFLOW_TRACKING_URI", None)
os.environ.pop("SENETRA_DB_PATH", None)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

HOLDBACK_DAYS = 3


def _load_ml_test_support():
    spec = importlib.util.spec_from_file_location("senetra_ml_test_support", REPO / "ml" / "tests" / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ml_support = _load_ml_test_support()


def make_settings(home: Path, **overrides):
    from mlops.settings import MLOpsSettings

    values = dict(
        state_db=home / "mlops.db", scheduler_enabled=False, job_runner="inline", retrain_targets=("ors",),
        retrain_cooldown_minutes=0, canary_stages=(50, 100), canary_min_stage_minutes=0, canary_min_requests=3,
        canary_max_percent_without_actuals=100, admin_token="secret",
    )
    values.update(overrides)
    return MLOpsSettings(**values)


def deliver_held_back_data(source: Path, live: Path, cutoff: str) -> None:
    """Simulates ingestion: copies the held-back days from the full database into the live one."""
    conn = sqlite3.connect(live)
    try:
        conn.execute("ATTACH DATABASE ? AS source", (str(source),))
        for table in ("daily_metrics", "medicine_consumption"):
            conn.execute(f"INSERT INTO {table} SELECT * FROM source.{table} WHERE date > ?", (cutoff,))
        conn.commit()
        conn.execute("DETACH DATABASE source")
    finally:
        conn.close()


@pytest.fixture(scope="module")
def late_arriving_db(tmp_path_factory):
    root = tmp_path_factory.mktemp("arrival")
    full = ml_support.build_synthetic_db(root / "full.db")
    live = root / "senetra.db"
    shutil.copy(full, live)
    conn = sqlite3.connect(live)
    try:
        (last,), = conn.execute("SELECT MAX(date) FROM daily_metrics").fetchall()
        cutoff = (pd.Timestamp(last) - pd.Timedelta(days=HOLDBACK_DAYS)).strftime("%Y-%m-%d")
        for table in ("daily_metrics", "medicine_consumption"):
            conn.execute(f"DELETE FROM {table} WHERE date > ?", (cutoff,))
        conn.commit()
    finally:
        conn.close()
    return full, live, cutoff
