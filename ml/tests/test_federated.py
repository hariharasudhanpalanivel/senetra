import numpy as np
import pytest

from senetra_ml.config import FederatedConfig
from senetra_ml.federated.glm import GLMProblem, newton_step
from senetra_ml.federated.trainer import HierarchicalFederatedTrainer, initial_params

DISTRICT_OF_CLIENT = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
COUNTRY_OF_DISTRICT = np.array([0, 0, 1])


def make_problem(seed: int = 0, rows: int = 300, district2_intercept: float = 2.0) -> tuple[GLMProblem, np.ndarray]:
    rng = np.random.default_rng(seed)
    beta = np.array([2.0, 0.3, -0.2, 0.1])
    X = rng.normal(size=(12, rows, beta.size))
    X[..., 0] = 1.0
    coefficients = np.repeat(beta[None], 12, axis=0)
    coefficients[DISTRICT_OF_CLIENT == 2, 0] = district2_intercept
    y = rng.poisson(np.exp(np.einsum("cnf,cf->cn", X, coefficients))).astype(float)
    mask = np.ones((12, rows), dtype=bool)
    mask[0, rows // 2:] = False  # unequal client sizes
    return GLMProblem.build(X, y, mask, "poisson", l2=0.0), beta


def fit(problem: GLMProblem, cfg: FederatedConfig):
    return HierarchicalFederatedTrainer(cfg).fit(
        problem, initial_params(problem), phc_ids=np.arange(12), district_ids=np.arange(3),
        district_index=DISTRICT_OF_CLIENT, country_ids=np.arange(2), district_country_index=COUNTRY_OF_DISTRICT,
    )


def centralized(problem: GLMProblem, clients: np.ndarray | None = None, steps: int = 25) -> np.ndarray:
    idx = np.arange(problem.X.shape[0]) if clients is None else clients
    F = problem.X.shape[-1]
    pooled = GLMProblem.build(problem.X[idx].reshape(1, -1, F), problem.y[idx].reshape(1, -1),
                              problem.w[idx].reshape(1, -1).astype(bool), problem.family, problem.l2)
    params = initial_params(pooled)[None]
    for _ in range(steps):
        params = newton_step(pooled, params, params, 0.0)
    return params[0]


def test_client_update_depends_only_on_its_own_rows():
    problem, _ = make_problem()
    params = np.zeros((12, 4))
    params[:, 0] = 1.5
    base = newton_step(problem, params, params, 0.1)
    X, y = problem.X.copy(), problem.y.copy()
    X[0, :, 1:] += 5.0
    y[0] *= 3.0
    changed = newton_step(GLMProblem(X, y, problem.w, "poisson", 0.0), params, params, 0.1)
    np.testing.assert_allclose(base[1:], changed[1:])
    assert not np.allclose(base[0], changed[0])


def test_federated_newton_world_model_equals_centralized_fit():
    problem, _ = make_problem(seed=1, district2_intercept=2.4)
    model = fit(problem, FederatedConfig(rounds=15, l2=0.0, level_newton_steps=0, personalization_steps=0))
    np.testing.assert_allclose(model.world, centralized(problem), atol=1e-6)
    objectives = [h["world_objective"] for h in model.history]
    assert all(b <= a + 1e-12 for a, b in zip(objectives, objectives[1:]))


def test_levels_without_prior_recover_each_group_fit():
    problem, _ = make_problem(seed=2, district2_intercept=2.6)
    model = fit(problem, FederatedConfig(rounds=15, l2=0.0, level_newton_steps=20, country_prior_rows=0,
                                         district_prior_rows=0, personalization_steps=0))
    district2 = centralized(problem, np.flatnonzero(DISTRICT_OF_CLIENT == 2))
    np.testing.assert_allclose(model.district[2], district2, atol=1e-5)
    np.testing.assert_allclose(model.country[1], district2, atol=1e-5)  # country 1 contains only district 2
    assert model.district[2][0] > model.world[0] + 0.1


def test_strong_priors_keep_every_level_at_its_parent():
    problem, _ = make_problem(seed=3, district2_intercept=2.6)
    model = fit(problem, FederatedConfig(rounds=15, l2=0.0, country_prior_rows=1e9, district_prior_rows=1e9,
                                         phc_prior_rows=1e9, personalization_steps=3))
    np.testing.assert_allclose(model.country, np.repeat(model.world[None], 2, axis=0), atol=1e-4)
    np.testing.assert_allclose(model.district, np.repeat(model.world[None], 3, axis=0), atol=1e-4)
    np.testing.assert_allclose(model.phc, model.district[DISTRICT_OF_CLIENT], atol=1e-4)


def test_personalization_without_prior_matches_local_fit():
    problem, _ = make_problem(seed=4)
    cfg = FederatedConfig(rounds=10, l2=0.0, phc_prior_rows=0, personalization_steps=15, local_only_steps=15)
    model = fit(problem, cfg)
    local = HierarchicalFederatedTrainer(cfg).fit_local_only(problem, initial_params(problem))
    np.testing.assert_allclose(model.phc, local, atol=1e-4)


@pytest.mark.parametrize("algorithm", ["newton", "fedavg"])
def test_gaussian_family_recovers_large_mean_targets(algorithm):
    rng = np.random.default_rng(7)
    beta = np.array([60.0, 5.0, -3.0, 0.0])
    X = rng.normal(size=(12, 200, 4))
    X[..., 0] = 1.0
    y = X @ beta + rng.normal(0, 1.0, size=(12, 200))
    problem = GLMProblem.build(X, y, np.ones((12, 200), dtype=bool), "gaussian", l2=0.0)
    model = fit(problem, FederatedConfig(algorithm=algorithm, rounds=10, l2=0.0, proximal_mu=0.0,
                                         personalization_steps=0))
    np.testing.assert_allclose(model.world, beta, atol=0.1)


def test_fedavg_round_equals_weighted_average_of_local_updates():
    problem, _ = make_problem()
    cfg = FederatedConfig(algorithm="fedavg", rounds=1, district_rounds=1, local_newton_steps=1, proximal_mu=0.0,
                          l2=0.0, personalization_steps=0)
    model = fit(problem, cfg)
    start = np.repeat(initial_params(problem)[None], 12, axis=0)
    local = newton_step(problem, start, start, 0.0)
    expected = (local * problem.n[:, None]).sum(axis=0) / problem.n.sum()
    np.testing.assert_allclose(model.world, expected, rtol=1e-10)


def test_fedavg_converges_near_the_centralized_solution_on_iid_clients():
    problem, beta = make_problem(seed=5)
    model = fit(problem, FederatedConfig(algorithm="fedavg", rounds=20, proximal_mu=0.0, l2=0.0,
                                         personalization_steps=0))
    np.testing.assert_allclose(model.world, centralized(problem), atol=2e-2)
    np.testing.assert_allclose(model.world, beta, atol=5e-2)


def test_noise_requires_fedavg_with_clipping():
    with pytest.raises(ValueError):
        HierarchicalFederatedTrainer(FederatedConfig(algorithm="fedavg", dp_noise_multiplier=1.0))
    with pytest.raises(ValueError):
        HierarchicalFederatedTrainer(FederatedConfig(update_clip_norm=1.0, dp_noise_multiplier=1.0))


def test_clipped_noisy_fedavg_still_runs():
    problem, _ = make_problem(seed=6)
    model = fit(problem, FederatedConfig(algorithm="fedavg", rounds=3, update_clip_norm=1.0, dp_noise_multiplier=0.1))
    assert np.all(np.isfinite(model.world))
