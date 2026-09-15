import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from conftest import ml_support
from mlops.canary import VariantStats, decide
from mlops.data_watch import DataSignature, decide_retrain, read_signature
from mlops.deployment import TargetDeployment, traffic_bucket
from mlops.settings import MLOpsSettings
from mlops.state import StateStore

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
SETTINGS = MLOpsSettings(state_db=Path("unused.db"), canary_stages=(10, 25, 50, 100), canary_min_requests=20,
                         canary_min_stage_minutes=60, canary_max_percent_without_actuals=25)
HEALTHY = VariantStats(requests=100, errors=0, p95_latency_ms=100.0)
PENDING = {"status": "pending"}


def canary(stage: int = 0, minutes_ago: float = 120) -> TargetDeployment:
    return TargetDeployment(
        target="ors", model_name="senetra-forecast-ors", stable_version="1", canary_version="2",
        canary_percent=SETTINGS.canary_stages[stage], canary_stage_index=stage,
        canary_stage_started_at=(NOW - timedelta(minutes=minutes_ago)).isoformat(),
    )


def compared(stable: float, candidate: float) -> dict:
    return {"status": "compared", "stable_district_wape": stable, "canary_district_wape": candidate}


# ---------------------------------------------------------------------------- canary decisions

@pytest.mark.parametrize(("deployment", "stats", "accuracy", "action"), [
    (canary(0), VariantStats(5, 0, 90.0), PENDING, "hold"),                 # not enough requests
    (canary(0, minutes_ago=10), HEALTHY, PENDING, "hold"),                   # stage too young
    (canary(0), VariantStats(50, 5, 100.0), PENDING, "rollback"),            # 10% errors
    (canary(0), VariantStats(50, 0, 900.0), PENDING, "rollback"),            # 9x slower
    (canary(0), VariantStats(50, 0, 200.0), PENDING, "advance"),             # slower but under the latency floor
    (canary(1), HEALTHY, compared(0.040, 0.060), "rollback"),                # accuracy regression
    (canary(0), HEALTHY, PENDING, "advance"),                                # 10% -> 25% allowed without actuals
    (canary(1), HEALTHY, PENDING, "hold"),                                   # 25% -> 50% needs fresh actuals
    (canary(1), HEALTHY, compared(0.040, 0.041), "advance"),                 # within tolerance
    (canary(3), HEALTHY, compared(0.040, 0.039), "promote"),                 # final stage, verified
    (canary(3), HEALTHY, PENDING, "hold"),                                   # final stage, unverified
])
def test_canary_decisions(deployment, stats, accuracy, action):
    decision = decide(deployment, stats, HEALTHY, accuracy, SETTINGS, NOW)
    assert decision.action == action, decision.reason


def test_traffic_split_is_sticky_and_proportional():
    keys = [f"client-{i}" for i in range(4000)]
    share = sum(traffic_bucket("ors", "7", key) < 25 for key in keys) / len(keys)
    assert 0.22 < share < 0.28
    assert [traffic_bucket("ors", "7", k) for k in keys[:50]] == [traffic_bucket("ors", "7", k) for k in keys[:50]]
    assert any(traffic_bucket("ors", "8", k) != traffic_bucket("ors", "7", k) for k in keys[:50])  # reshuffled per version


# ---------------------------------------------------------------------------- retraining policy

def signature(date: str, rows: int) -> DataSignature:
    return DataSignature(latest_date=date, tables={"daily_metrics": {"rows": rows, "max_date": date}})


POLICY = MLOpsSettings(state_db=Path("unused.db"), retrain_min_new_days=2, retrain_cooldown_minutes=60)


@pytest.mark.parametrize(("current", "trained", "kwargs", "retrain"), [
    (signature("2026-09-10", 10), signature("2026-09-10", 10), {"force": True}, True),
    (signature("2026-09-10", 10), None, {}, True),
    (signature("2026-09-10", 10), signature("2026-09-10", 10), {}, False),
    (signature("2026-09-13", 40), signature("2026-09-10", 10), {"canary_active": True}, False),
    (signature("2026-09-13", 40), signature("2026-09-10", 10), {"last_finished_at": NOW - timedelta(minutes=10)}, False),
    (signature("2026-09-11", 20), signature("2026-09-10", 10), {}, False),   # 1 day < 2 required
    (signature("2026-09-13", 40), signature("2026-09-10", 10), {}, True),
    (signature("2026-09-10", 12), signature("2026-09-10", 10), {}, True),    # backfill / corrections
])
def test_retraining_policy(current, trained, kwargs, retrain):
    options = {"last_finished_at": None, "canary_active": False, "force": False, **kwargs}
    decision = decide_retrain(current, trained, POLICY, now=NOW, **options)
    assert decision.should_retrain is retrain, decision.reason


def test_signature_sees_new_training_data_but_ignores_inventory(tmp_path):
    db = ml_support.build_synthetic_db(tmp_path / "senetra.db")
    before = read_signature(db)
    conn = sqlite3.connect(db)
    try:
        conn.execute("UPDATE inventory SET current_stock = current_stock + 1")
        conn.commit()
        assert read_signature(db).to_json() == before.to_json()
        conn.execute("INSERT INTO daily_metrics (phc_id, date, patient_footfall, bed_occupancy, staff_availability,"
                     " outbreak_flag) VALUES (1, '2099-01-01', 10, 50, 90, 0)")
        conn.commit()
    finally:
        conn.close()
    after = read_signature(db)
    assert after.latest_date == "2099-01-01" and after.to_json() != before.to_json()


# ---------------------------------------------------------------------------- settings and state

def test_settings_from_environment():
    settings = MLOpsSettings.from_env({"ML_CANARY_STAGES": "5,50,100", "ML_SCHEDULER_ENABLED": "false",
                                       "ML_RETRAIN_TARGETS": "ors, paracetamol", "ML_ADMIN_TOKEN": "t"})
    assert settings.canary_stages == (5, 50, 100) and settings.scheduler_enabled is False
    assert settings.retrain_targets == ("ors", "paracetamol") and settings.admin_token == "t"
    with pytest.raises(ValueError):
        MLOpsSettings(canary_stages=(50, 10))


def test_scheduler_lease_has_a_single_owner(tmp_path):
    state = StateStore(tmp_path / "state.db")
    assert state.try_acquire_lease("scheduler", "a", 60)
    assert not state.try_acquire_lease("scheduler", "b", 60)
    assert state.try_acquire_lease("scheduler", "a", 60)
    state.release_lease("scheduler", "a")
    assert state.try_acquire_lease("scheduler", "b", 60)
    assert state.try_acquire_lease("expired", "a", -1)
    assert state.try_acquire_lease("expired", "b", 60)


def test_prediction_stats_exclude_forced_and_client_errors(tmp_path):
    state = StateStore(tmp_path / "state.db")
    common = dict(target="ors", variant="canary", model_version="2", level="district", entity_id=1, scenario="observed")
    for latency in (10.0, 20.0, 30.0):
        state.log_prediction(status="ok", latency_ms=latency, **common)
    state.log_prediction(status="error", latency_ms=None, **common)
    state.log_prediction(status="client_error", latency_ms=5.0, **common)
    state.log_prediction(status="error", latency_ms=None, forced=True, **common)
    stats = state.prediction_stats("ors", "canary", "2", since="")
    assert stats["requests"] == 4 and stats["errors"] == 1
    assert 20.0 < stats["p95_latency_ms"] <= 30.0
