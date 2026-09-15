"""End-to-end through the Flask API: bootstrap -> new data -> retrain -> canary -> promote -> rollback -> revert.

Tests in this module share one app and run in order, like a deployment timeline.
"""

from types import SimpleNamespace

import pytest

from conftest import deliver_held_back_data, make_settings, ml_support

AUTH = {"Authorization": "Bearer secret"}
FORECAST = "/api/ml/forecast/ors?level=district&entity_id=1"


@pytest.fixture(scope="module")
def ctx(tmp_path_factory, late_arriving_db):
    full, live, cutoff = late_arriving_db
    home = tmp_path_factory.mktemp("home")
    from app import create_app

    app = create_app({"TESTING": True, "MLOPS_SETTINGS": make_settings(home),
                      "MLOPS_PIPELINE_CONFIG": ml_support.make_cfg(home, live)})
    return SimpleNamespace(app=app, client=app.test_client(), full=full, live=live, cutoff=cutoff)


def deployment(ctx) -> dict:
    return ctx.client.get("/api/ml/deployments/ors").get_json()


def test_state_changing_endpoints_require_the_admin_token(ctx):
    assert ctx.client.post("/api/ml/retrain").status_code == 401
    assert ctx.client.post("/api/ml/data/arrived").status_code == 401
    assert ctx.client.post("/api/ml/deployments/ors/promote", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_forecasts_are_unavailable_before_any_model(ctx):
    assert ctx.client.get(FORECAST).status_code == 503
    status = ctx.client.get("/api/ml/data/status").get_json()
    assert status["would_retrain"] is True and status["last_trained"] is None


def test_first_data_check_bootstraps_a_champion(ctx):
    response = ctx.client.post("/api/ml/data/arrived", headers=AUTH)
    body = response.get_json()
    assert response.status_code == 202 and body["action"] == "retrain"
    assert body["job"]["status"] == "succeeded", body["job"]
    assert body["job"]["result"]["deployments"]["ors"]["action"] == "promoted"
    state = deployment(ctx)
    assert state["stable_version"] == "1" and state["canary_version"] is None


def test_forecasts_are_served_by_the_stable_model(ctx):
    response = ctx.client.get(FORECAST + "&scenario=normal")
    body = response.get_json()
    assert response.status_code == 200 and len(body["forecast"]) == 7
    assert body["deployment"] == {"variant": "stable", "model_version": "1", "stable_version": "1",
                                  "canary_version": None, "canary_percent": 0}
    assert ctx.client.get("/api/ml/forecast/not_a_target").status_code == 404
    assert ctx.client.get(FORECAST + "&scenario=outbreak").status_code == 422
    assert ctx.client.get("/api/ml/forecast/ors?level=district&entity_id=999").status_code == 404


def test_no_retraining_without_new_data(ctx):
    body = ctx.client.post("/api/ml/data/arrived", headers=AUTH).get_json()
    assert body["action"] == "none" and "no new data" in body["reason"]


def test_new_data_triggers_retraining_into_a_canary(ctx):
    assert ctx.client.get("/api/ml/data/status").get_json()["would_retrain"] is False
    deliver_held_back_data(ctx.full, ctx.live, ctx.cutoff)
    status = ctx.client.get("/api/ml/data/status").get_json()
    assert status["would_retrain"] is True and "3 new day" in status["reason"]

    body = ctx.client.post("/api/ml/data/arrived", headers=AUTH).get_json()
    assert body["action"] == "retrain" and body["job"]["status"] == "succeeded", body
    assert body["job"]["result"]["deployments"]["ors"]["action"] == "canary"
    state = deployment(ctx)
    assert (state["stable_version"], state["canary_version"], state["canary_percent"]) == ("1", "2", 50)


def test_traffic_is_split_sticky_between_stable_and_canary(ctx):
    variants = {}
    for i in range(30):
        response = ctx.client.get(FORECAST, headers={"X-Client-Id": f"user-{i}"})
        assert response.status_code == 200
        variants[i] = response.get_json()["deployment"]["variant"]
    assert set(variants.values()) == {"stable", "canary"}
    repeat = ctx.client.get(FORECAST, headers={"X-Client-Id": "user-0"}).get_json()
    assert repeat["deployment"]["variant"] == variants[0]
    forced = ctx.client.get(FORECAST, headers={"X-ML-Variant": "canary"}).get_json()
    assert forced["deployment"]["model_version"] == "2"


def test_healthy_canary_advances_and_is_promoted(ctx):
    first = ctx.client.post("/api/ml/deployments/ors/evaluate", headers=AUTH).get_json()
    assert first["action"] == "advance", first["reason"]
    assert first["checks"]["accuracy"]["status"] == "pending"  # no actuals after the canary's training data yet
    assert deployment(ctx)["canary_percent"] == 100

    for i in range(4):
        body = ctx.client.get(FORECAST, headers={"X-Client-Id": f"late-{i}"}).get_json()
        assert body["deployment"]["variant"] == "canary"
    second = ctx.client.post("/api/ml/deployments/ors/evaluate", headers=AUTH).get_json()
    assert second["action"] == "promote", second["reason"]

    state = deployment(ctx)
    assert (state["stable_version"], state["previous_version"], state["canary_version"]) == ("2", "1", None)
    served = ctx.client.get(FORECAST).get_json()["deployment"]
    assert (served["variant"], served["model_version"]) == ("stable", "2")


def test_failing_canary_is_rolled_back(ctx):
    body = ctx.client.post("/api/ml/retrain", json={"force": True}, headers=AUTH).get_json()
    assert body["job"]["status"] == "succeeded", body
    assert deployment(ctx)["canary_version"] == "3"

    state = ctx.app.extensions["senetra_mlops"].state
    for _ in range(5):
        state.log_prediction(target="ors", variant="canary", model_version="3", level="district", entity_id=1,
                             scenario="observed", status="error", latency_ms=None)
    result = ctx.client.post("/api/ml/deployments/ors/evaluate", headers=AUTH).get_json()
    assert result["action"] == "rollback" and "error rate" in result["reason"]

    after = deployment(ctx)
    assert after["canary_version"] is None and after["stable_version"] == "2"
    assert ctx.client.get(FORECAST, headers={"X-Client-Id": "user-1"}).get_json()["deployment"]["model_version"] == "2"


def test_revert_restores_the_previous_champion(ctx):
    body = ctx.client.post("/api/ml/deployments/ors/revert", json={"reason": "test"}, headers=AUTH).get_json()
    assert body["version"] == "1"
    state = deployment(ctx)
    assert (state["stable_version"], state["previous_version"]) == ("1", "2")


def test_jobs_events_and_health_are_reported(ctx):
    jobs = ctx.client.get("/api/ml/jobs").get_json()
    assert [job["status"] for job in jobs] == ["succeeded"] * 3
    assert ctx.client.get(f"/api/ml/jobs/{jobs[0]['id']}").status_code == 200
    actions = {event["action"] for event in deployment(ctx)["events"]}
    assert {"bootstrap_promote", "canary_start", "canary_advance", "promote", "rollback", "revert"} <= actions
    health = ctx.client.get("/api/ml/health").get_json()
    ors = next(d for d in health["deployments"] if d["target"] == "ors")
    assert ors["stable_version"] == "1" and health["scheduler"]["enabled"] is False
