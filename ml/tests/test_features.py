import copy

import numpy as np
import pandas as pd
import pytest

from senetra_ml.data.panel import load_panel
from senetra_ml.data.repository import SenetraRepository
from senetra_ml.features.engineering import BASELINES, build_features, trailing_stats


@pytest.fixture
def panel(cfg):
    with SenetraRepository(cfg.data.db_path) as repo:
        return load_panel(repo)


def test_trailing_stats_match_pandas_rolling():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 40))
    x[rng.random(x.shape) < 0.1] = np.nan
    mean, std = trailing_stats(x, window=7, min_periods=4)
    for row in range(3):
        rolling = pd.Series(x[row]).rolling(7, min_periods=4)
        np.testing.assert_allclose(mean[row], rolling.mean().to_numpy(), equal_nan=True)
        np.testing.assert_allclose(std[row], rolling.std(ddof=0).to_numpy(), equal_nan=True, atol=1e-9)


def test_panel_reads_hierarchy_and_prevalence(panel):
    assert panel.n_phcs == 11
    assert list(panel.country_ids) == [1, 2]
    assert panel.district_country_index.tolist() == [0, 0, 1]
    prevalence = panel.district_prevalence
    assert np.nanmax(prevalence[:, 35:47]) > 0 and np.nanmax(prevalence[:, :35]) == 0


def test_history_features_never_see_the_future(cfg, panel):
    origin = 40
    fs = build_features(panel, "ors", cfg, origins=np.array([origin]))
    tampered = copy.deepcopy(panel)
    tampered.values[:, origin + 1:, :] *= 3.0
    tampered.district_prevalence[:, origin + 1:] = 0.9
    fs_tampered = build_features(tampered, "ors", cfg, origins=np.array([origin]))

    history = [i for i in range(len(fs.names)) if i != fs.scenario_index]
    np.testing.assert_allclose(fs.X[..., history], fs_tampered.X[..., history])
    for name in BASELINES:
        np.testing.assert_allclose(fs.baselines[name], fs_tampered.baselines[name])
    assert not np.allclose(fs.y, fs_tampered.y), "targets are future values and must change"


def test_train_mask_only_uses_targets_up_to_cutoff(cfg, panel):
    fs = build_features(panel, "paracetamol", cfg)
    cutoff = 30
    mask = fs.train_mask(cutoff)
    assert mask.any()
    assert np.broadcast_to(fs.target_idx, mask.shape)[mask].max() <= cutoff


def test_recent_outbreak_flags_mark_the_recovery_window(cfg, panel):
    # Synthetic outbreak covers days 35-46; the long window is 28 days.
    fs = build_features(panel, "ors", cfg, origins=np.array([20, 40, 60]))
    recent = fs.origin_recent_outbreak[:, fs.horizon == 1]
    assert not recent[:, 0].any()
    assert recent[:, 1].all() and recent[:, 2].all()


def test_scenario_covariate_is_the_target_day_outbreak_regime(cfg, panel):
    fs = build_features(panel, "ors", cfg)
    scenario = fs.X[..., fs.scenario_index]
    known = np.isfinite(scenario)
    assert set(np.unique(scenario[known])) <= {0.0, 1.0}
    expected = fs.target_prevalence > cfg.data.outbreak_prevalence_threshold
    np.testing.assert_array_equal(scenario[known] == 1.0, expected[known])
    assert np.isnan(scenario[:, fs.target_idx >= panel.n_dates]).all()
