"""Synthetic SENETRA database built from the real DDL, so tests never touch operational data."""

from __future__ import annotations

import importlib.util
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.pop("MLFLOW_TRACKING_URI", None)
os.environ.pop("SENETRA_DB_PATH", None)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from senetra_ml.config import PipelineConfig, load_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
N_DAYS = 70
OUTBREAK_DAYS = range(35, 47)
MEDICINES = {"ORS": (40.0, 2.0), "Paracetamol": (70.0, 1.8), "IV Fluids": (10.0, 1.5),
             "Antibiotics": (27.0, 1.0), "Insulin": (3.5, 1.0)}
PHC_DISTRICT = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}


def _sqlite_ddl() -> str:
    spec = importlib.util.spec_from_file_location("build_sqlite", REPO_ROOT / "seeds" / "build_sqlite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.translate_ddl((REPO_ROOT / "seeds" / "DDL.sql").read_text(encoding="utf-8"))


def build_synthetic_db(path: Path) -> Path:
    rng = np.random.default_rng(7)
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(end - pd.Timedelta(days=N_DAYS - 1), end)
    conn = sqlite3.connect(path)
    conn.executescript(_sqlite_ddl())
    conn.executemany("INSERT INTO countries (id, name, code) VALUES (?, ?, ?)", [(1, "India", "IN"), (2, "Brazil", "BR")])
    conn.executemany("INSERT INTO states (id, country_id, name) VALUES (?, ?, ?)", [(1, 1, "Tamil Nadu"), (2, 2, "Sao Paulo")])
    conn.executemany("INSERT INTO districts (id, state_id, name) VALUES (?, ?, ?)",
                     [(1, 1, "Chennai"), (2, 1, "Madurai"), (3, 2, "Campinas")])
    conn.executemany("INSERT INTO phcs (id, district_id, name, total_beds, total_doctors, total_nurses) VALUES (?, ?, ?, 30, 5, 12)",
                     [(phc, district, f"PHC {phc}") for phc, district in PHC_DISTRICT.items()])
    conn.executemany("INSERT INTO medicines (id, name, category, unit) VALUES (?, ?, 'test', 'unit')",
                     [(i, name) for i, name in enumerate(MEDICINES, start=1)])

    metrics, consumption, inventory = [], [], []
    for phc in PHC_DISTRICT:
        for t, date in enumerate(dates):
            outbreak = t in OUTBREAK_DAYS
            flag = int(outbreak and rng.random() < 0.6)
            metrics.append((phc, date.strftime("%Y-%m-%d"), int(rng.poisson(90 * (1.8 if outbreak else 1.0))),
                            float(np.clip(rng.normal(80 if outbreak else 57, 6), 0, 100)),
                            float(rng.uniform(80, 100)), flag, "DENGUE" if flag else None))
            for medicine_id, (base, uplift) in enumerate(MEDICINES.values(), start=1):
                used = float(rng.poisson(base * (uplift if outbreak else 1.0)))
                consumption.append((phc, medicine_id, date.strftime("%Y-%m-%d"), 1000.0, used, 1000.0 - used))
        for medicine_id, (base, _) in enumerate(MEDICINES.values(), start=1):
            inventory.append((phc, medicine_id, float(rng.integers(20, 40) * base), base, end.strftime("%Y-%m-%d")))

    conn.executemany("INSERT INTO daily_metrics (phc_id, date, patient_footfall, bed_occupancy, staff_availability,"
                     " outbreak_flag, outbreak_type) VALUES (?, ?, ?, ?, ?, ?, ?)", metrics)
    conn.executemany("INSERT INTO medicine_consumption (phc_id, medicine_id, date, opening_stock, consumption,"
                     " closing_stock) VALUES (?, ?, ?, ?, ?, ?)", consumption)
    conn.executemany("INSERT INTO inventory (phc_id, medicine_id, current_stock, daily_consumption, updated_at)"
                     " VALUES (?, ?, ?, ?, ?)", inventory)
    conn.execute("INSERT INTO simulation_events (event_type, district_id, severity, active, started_at)"
                 " VALUES ('DENGUE', 1, 8, 1, ?)", ((end - pd.Timedelta(days=5)).strftime("%Y-%m-%d 00:00:00"),))
    conn.commit()
    conn.close()
    return path


def make_cfg(home: Path, db_path: Path) -> PipelineConfig:
    base = load_config().model_dump()
    base["home"] = home
    base["data"]["db_path"] = db_path
    base["federated"]["rounds"] = 8
    base["benchmarks"]["xgboost"].update(num_boost_round=40, early_stopping_rounds=10)
    base["backtest"].update(n_folds=3, step_days=14)
    base["evaluation"]["gates"].update(max_district_wape=1.0, min_phc_skill_vs_baseline=-1.0,
                                       district_coverage80=(0.0, 1.0))
    base["mlflow"]["tracking_uri"] = f"sqlite:///{(home / 'mlruns' / 'mlflow.db').as_posix()}"
    return PipelineConfig.model_validate(base)


@pytest.fixture(scope="session")
def synthetic_db(tmp_path_factory) -> Path:
    return build_synthetic_db(tmp_path_factory.mktemp("db") / "senetra.db")


@pytest.fixture
def cfg(tmp_path, synthetic_db) -> PipelineConfig:
    return make_cfg(tmp_path, synthetic_db)
