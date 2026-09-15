import math

import numpy as np
import pandas as pd
import pytest

from senetra_ml.config import RiskConfig
from senetra_ml.evaluation.calibration import interval_bounds, quantile_table
from senetra_ml.evaluation.metrics import aggregate_to_level, skill, summarize_errors
from senetra_ml.monitoring.drift import bin_counts, psi
from senetra_ml.risk import days_of_supply, risk_level


def test_summarize_errors_matches_hand_computation():
    frame = pd.DataFrame({"actual": [10.0, 20.0, 30.0], "model": [12.0, 18.0, 33.0], "g": ["a", "a", "b"]})
    row = summarize_errors(frame, ["model"]).iloc[0]
    assert row["mae"] == pytest.approx(7 / 3)
    assert row["rmse"] == pytest.approx(math.sqrt(17 / 3))
    assert row["wape"] == pytest.approx(7 / 60)
    assert row["bias"] == pytest.approx(3 / 60)
    grouped = summarize_errors(frame, ["model"], by=["g"]).set_index("g")
    assert grouped.loc["b", "mae"] == pytest.approx(3.0)
    assert skill(1.0, 2.0) == pytest.approx(0.5)


def test_aggregate_to_level_sums_counts_and_averages_percentages():
    frame = pd.DataFrame({
        "phc_id": [1, 2, 3], "district_id": [1, 1, 2], "country_id": [1, 1, 1],
        "fold": 0, "origin_date": "2026-01-01", "horizon": 1, "target_date": "2026-01-02",
        "actual": [10.0, 20.0, 40.0], "outbreak": [True, True, False], "transition": False, "cold_start": False,
    })
    district = aggregate_to_level(frame, "district", ["actual"], "sum").set_index("district_id")
    assert district.loc[1, "actual"] == 30 and district.loc[1, "n_phcs"] == 2 and district.loc[1, "outbreak"]
    world = aggregate_to_level(frame, "world", ["actual"], "mean")
    assert world["actual"].iloc[0] == pytest.approx(70 / 3)


def test_conformal_intervals_cover_nominal_share():
    rng = np.random.default_rng(0)
    pred = rng.uniform(50, 150, 20000)
    actual = pred * (1 + rng.normal(0, 0.1, pred.size))
    horizon = rng.integers(1, 8, pred.size)
    calib, test = slice(0, 10000), slice(10000, None)
    table = quantile_table(actual[calib], pred[calib], horizon[calib])
    bounds = interval_bounds(pred[test], horizon[test], table)
    coverage80 = np.mean((actual[test] >= bounds["lo80"]) & (actual[test] <= bounds["hi80"]))
    coverage95 = np.mean((actual[test] >= bounds["lo95"]) & (actual[test] <= bounds["hi95"]))
    assert coverage80 == pytest.approx(0.8, abs=0.02)
    assert coverage95 == pytest.approx(0.95, abs=0.015)


@pytest.mark.parametrize(("stock", "daily", "expected"), [
    (0, [10, 10], 0.0),
    (25, [10, 10, 10], 2.5),
    (30, [10, 10, 10], 3.0),
    (50, [10, 10], 5.0),
])
def test_days_of_supply(stock, daily, expected):
    assert days_of_supply(stock, daily) == pytest.approx(expected)


def test_days_of_supply_with_no_demand_is_unbounded():
    assert days_of_supply(10, [0, 0]) == math.inf
    assert risk_level(math.inf, RiskConfig()) == "HEALTHY"
    assert risk_level(None, RiskConfig()) == "UNKNOWN"


@pytest.mark.parametrize(("days", "level"), [(2, "CRITICAL"), (3, "CRITICAL"), (6.5, "HIGH"), (10, "WATCH"), (20, "HEALTHY")])
def test_risk_levels(days, level):
    assert risk_level(days, RiskConfig()) == level


def test_psi_detects_shift_but_not_resampling():
    rng = np.random.default_rng(0)
    edges = np.linspace(-3, 3, 11)
    reference = bin_counts(rng.normal(size=50000), edges)
    expected = reference / reference.sum()
    assert psi(expected, bin_counts(rng.normal(size=5000), edges)) < 0.02
    assert psi(expected, bin_counts(rng.normal(1.0, 1.0, size=5000), edges)) > 0.25
