"""End-to-end: train -> evaluate -> forecast -> write -> monitor -> API on the synthetic database."""

import shutil
import sqlite3

import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from senetra_ml.config import LEVELS
from senetra_ml.data.repository import SenetraRepository
from senetra_ml.data.validation import validate_database
from senetra_ml.inference import ForecastService, Scenario
from senetra_ml.pipelines.eda import run_eda
from senetra_ml.pipelines.evaluate import run_evaluation
from senetra_ml.pipelines.monitor import run_monitoring
from senetra_ml.pipelines.train import run_training
from senetra_ml.serving.api import create_app
from senetra_ml.tracking import setup_mlflow

from conftest import make_cfg

TARGETS = ["ors", "bed_occupancy"]
RISK_LEVELS = {"CRITICAL", "HIGH", "WATCH", "HEALTHY"}


@pytest.fixture(scope="module")
def trained(tmp_path_factory, synthetic_db):
    cfg = make_cfg(tmp_path_factory.mktemp("home"), synthetic_db)
    training = run_training(cfg, TARGETS)
    evaluation = run_evaluation(cfg, training["run_id"])
    return cfg, training, evaluation


def test_validation_passes_on_synthetic_data(cfg):
    with SenetraRepository(cfg.data.db_path) as repo:
        report = validate_database(repo, cfg)
    assert not report.errors


def test_eda_report(cfg):
    summary = run_eda(cfg, track=False)
    assert (cfg.reports_dir / "eda" / "eda_report.md").is_file()
    assert summary["regimes"]["outbreak_days"] == 12
    assert summary["targets"]["ors"]["outbreak_uplift"] > 1.6
    assert summary["implications"]


def test_training_registers_models_without_logging_actuals(trained):
    cfg, training, _ = trained
    setup_mlflow(cfg)
    for target in TARGETS:
        result = training["targets"][target]
        assert result["model_version"] is not None and result["selected_level"] in LEVELS
        path = mlflow.artifacts.download_artifacts(run_id=result["run_id"], artifact_path="backtest/predictions.parquet")
        columns = set(pd.read_parquet(path).columns)
        assert "actual" not in columns
        assert {"fl_phc__observed", "fl_world__oracle", "local__observed", "xgb__oracle", "bl_mean7"} <= columns
    assert training["targets"]["ors"]["outbreak_effect"] > 1.5


def test_evaluation_promotes_champions_and_reports_cold_start(trained):
    cfg, _, evaluation = trained
    client = setup_mlflow(cfg)
    for result in evaluation["results"]:
        assert result["gates_passed"]
        champion = client.get_model_version_by_alias(cfg.registered_model_name(result["target"]), "champion")
        assert str(champion.version) == str(result["model_version"])
        assert np.isfinite(result["district_wape_cold_start_outbreak"])
    ors = next(r for r in evaluation["results"] if r["target"] == "ors")
    assert ors["district_wape_outbreak_scenario"] < ors["district_wape_cold_start_outbreak"]


def test_forecasts_at_every_level_and_scenario(trained):
    cfg, *_ = trained
    setup_mlflow(cfg)
    service = ForecastService(cfg)
    for level, entity in (("phc", 1), ("district", 1), ("country", 2), ("world", None)):
        result = service.forecast("ors", level, entity)
        assert len(result["forecast"]) == cfg.forecast.horizon_days
        assert result["stock"]["risk_level"] in RISK_LEVELS
        for point in result["forecast"]:
            assert point["lower_95"] <= point["lower_80"] <= point["upper_80"] <= point["upper_95"]

    normal = service.forecast("ors", "district", 1, Scenario("normal"))
    outbreak = service.forecast("ors", "district", 1, Scenario("outbreak", severity=8))
    assert outbreak["horizon_total"] > 1.5 * normal["horizon_total"]
    scoped = service.forecast("ors", "district", 3, Scenario("outbreak", severity=8, district_ids=(1,)))
    assert scoped["horizon_total"] < 1.2 * service.forecast("ors", "district", 3, Scenario("normal"))["horizon_total"]
    simulated = service.forecast("ors", "district", 1, Scenario("simulated"))
    assert simulated["horizon_total"] > 1.5 * normal["horizon_total"]

    world = service.forecast("ors", "world", None, Scenario("normal"))["horizon_total"]
    countries = sum(service.forecast("ors", "country", c, Scenario("normal"))["horizon_total"] for c in (1, 2))
    assert world == pytest.approx(countries, rel=1e-3)

    beds = service.forecast("bed_occupancy", "district", 2, Scenario("outbreak", severity=8))
    assert all(0 <= p["yhat"] <= 100 for p in beds["forecast"])
    normal_beds = service.forecast("bed_occupancy", "district", 2, Scenario("normal"))
    mean_outbreak = np.mean([p["yhat"] for p in beds["forecast"]])
    mean_normal = np.mean([p["yhat"] for p in normal_beds["forecast"]])
    assert 70 < mean_outbreak < 90 and 50 < mean_normal < 65  # synthetic truth: ~80 vs ~57
    with pytest.raises(LookupError):
        service.forecast("ors", "district", 999)


def test_write_predictions_is_idempotent(trained, tmp_path):
    cfg, *_ = trained
    db_copy = tmp_path / "senetra.db"
    shutil.copy(cfg.data.db_path, db_copy)
    local_cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"db_path": db_copy})})
    setup_mlflow(local_cfg)
    service = ForecastService(local_cfg)
    service.write_predictions("ors", Scenario())
    written = service.write_predictions("ors", Scenario())
    with sqlite3.connect(db_copy) as conn:
        (rows,), = conn.execute("SELECT COUNT(*) FROM predictions").fetchall()
    assert written == rows == 11 * cfg.forecast.horizon_days


def test_monitoring_reports_drift_and_performance(trained):
    cfg, *_ = trained
    report = run_monitoring(cfg, ["ors"])
    assert report["status"] in {"ok", "warn", "alert"}
    ors = report["targets"]["ors"]
    assert ors["status"] == "checked" and ors["performance"]["rows"] > 0
    assert (cfg.reports_dir / "monitoring" / "latest.json").is_file()


def test_api_endpoints(trained):
    cfg, *_ = trained
    client = TestClient(create_app(cfg))
    health = client.get("/health").json()
    assert health["models"]["ors"] and health["models"]["insulin"] is None and health["status"] == "degraded"

    response = client.get("/v1/forecast/ors", params={"level": "district", "entity_id": 1,
                                                       "scenario": "outbreak", "severity": 8})
    assert response.status_code == 200 and len(response.json()["forecast"]) == cfg.forecast.horizon_days
    assert client.get("/v1/forecast/ors", params={"level": "district", "entity_id": 999}).status_code == 404
    assert client.get("/v1/forecast/ors", params={"level": "district", "entity_id": 1,
                                                  "scenario": "outbreak"}).status_code == 422
    assert client.get("/v1/forecast/not_a_target").status_code == 404
    everything = client.get("/v1/forecast", params={"level": "world"}).json()
    assert set(everything["forecasts"]) == set(TARGETS)
    federation = client.get("/v1/federation/ors").json()
    assert federation["privacy_contract"] and len(federation["hierarchy"]) == 2
    assert len(client.get("/v1/hierarchy").json()) == 2
