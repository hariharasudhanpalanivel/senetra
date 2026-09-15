"""Hierarchical federated training: PHC clients -> district -> country -> world.

Default algorithm "newton" (exact federated Newton, GLORE-style):
1. World model: each round, every PHC computes loss/gradient/Hessian sums of its own rows at the
   current world model; districts add them up, countries add district sums, the world adds country
   sums and takes one Newton step. The fixed point equals a centralized fit, without moving rows.
2. Country models refine the world model on country sums, district models refine their country
   model on district sums, PHC models refine their district model on their own rows. Each
   refinement is shrunk toward its parent with a prior worth `*_prior_rows` rows.

Alternative "fedavg" (FedAvg/FedProx): PHCs run local Newton steps and districts average parameter
deltas (`district_rounds` times per global round) before country/world averaging. It supports
update clipping and Gaussian noise, but converges to a biased point on heterogeneous clients.

Only parameters, row counts and summed statistics cross tier boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from senetra_ml.config import FederatedConfig
from senetra_ml.federated.glm import GLMProblem, group_newton, inverse_link, linear_predictor, newton_step, objective
from senetra_ml.federated.strategy import clip_update_norms, gaussian_mechanism, group_weighted_mean


@dataclass
class FederatedModel:
    world: np.ndarray  # (F,)
    country_ids: np.ndarray  # (G,)
    country: np.ndarray  # (G, F)
    district_ids: np.ndarray  # (D,)
    district: np.ndarray  # (D, F)
    phc_ids: np.ndarray  # (C,)
    phc: np.ndarray  # (C, F)
    history: list[dict] = field(default_factory=list)

    def client_params(self, level: str, district_index: np.ndarray,
                      district_country_index: np.ndarray) -> np.ndarray:
        if level == "world":
            return np.broadcast_to(self.world, (len(district_index), self.world.size))
        if level == "country":
            return self.country[district_country_index[district_index]]
        if level == "district":
            return self.district[district_index]
        if level == "phc":
            return self.phc
        raise ValueError(f"Unknown level {level!r}")


class HierarchicalFederatedTrainer:
    def __init__(self, cfg: FederatedConfig):
        if cfg.dp_noise_multiplier > 0 and (cfg.algorithm != "fedavg" or cfg.update_clip_norm is None):
            raise ValueError("dp_noise_multiplier requires algorithm 'fedavg' and update_clip_norm")
        self.cfg = cfg

    def fit(self, problem: GLMProblem, init: np.ndarray, *, phc_ids: np.ndarray, district_ids: np.ndarray,
            district_index: np.ndarray, country_ids: np.ndarray,
            district_country_index: np.ndarray) -> FederatedModel:
        fit = self._fit_newton if self.cfg.algorithm == "newton" else self._fit_fedavg
        world, country, district, history = fit(problem, init, district_index, district_country_index,
                                                len(district_ids), len(country_ids))
        phc = district[district_index].copy()
        if self.cfg.personalization_steps:
            n_clients = problem.X.shape[0]
            phc, _ = group_newton(problem, phc, np.arange(n_clients), n_clients,
                                  self.cfg.phc_prior_rows, self.cfg.personalization_steps)
        return FederatedModel(world=world, country_ids=country_ids, country=country, district_ids=district_ids,
                              district=district, phc_ids=phc_ids, phc=phc, history=history)

    def _fit_newton(self, problem, init, district_index, district_country_index, n_districts, n_countries):
        cfg = self.cfg
        n_clients = problem.X.shape[0]
        world, steps = group_newton(problem, init[None], np.zeros(n_clients, dtype=int), 1, 0.0, cfg.rounds)
        participants = int((problem.n > 0).sum())
        history = [{"round": s["step"], "world_objective": s["objective"], "participants": participants} for s in steps]
        country, _ = group_newton(problem, np.repeat(world, n_countries, axis=0),
                                  district_country_index[district_index], n_countries,
                                  cfg.country_prior_rows, cfg.level_newton_steps)
        district, _ = group_newton(problem, country[district_country_index], district_index, n_districts,
                                   cfg.district_prior_rows, cfg.level_newton_steps)
        return world[0], country, district, history

    def _fit_fedavg(self, problem, init, district_index, district_country_index, n_districts, n_countries):
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        n_clients = problem.X.shape[0]
        dp = cfg.dp_noise_multiplier > 0
        n = problem.n
        has_data = n > 0
        district_rows = np.bincount(district_index, weights=n, minlength=n_districts)
        country_rows = np.bincount(district_country_index, weights=district_rows, minlength=n_countries)

        world = init.astype(float).copy()
        country = np.repeat(world[None], n_countries, axis=0)
        district = np.repeat(world[None], n_districts, axis=0)
        history: list[dict] = []
        for round_no in range(1, cfg.rounds + 1):
            district = np.repeat(world[None], n_districts, axis=0)
            participants = 0
            for _ in range(cfg.district_rounds):
                selected = has_data & (rng.random(n_clients) < cfg.client_fraction)
                participants = int(selected.sum())
                start = district[district_index]
                local = start
                for _ in range(cfg.local_newton_steps):
                    local = newton_step(problem, local, anchor=start, mu_prox=cfg.proximal_mu)
                delta = local - start
                if cfg.update_clip_norm is not None:
                    delta = clip_update_norms(delta, cfg.update_clip_norm)
                weights = np.where(selected, 1.0 if dp else n, 0.0)
                mean_delta, _ = group_weighted_mean(delta, weights, district_index, n_districts)
                if dp:
                    counts = np.bincount(district_index, weights=selected.astype(float), minlength=n_districts)
                    mean_delta = gaussian_mechanism(mean_delta, counts, cfg.update_clip_norm,
                                                    cfg.dp_noise_multiplier, rng)
                district = district + np.nan_to_num(mean_delta)

            country_new, _ = group_weighted_mean(district, district_rows, district_country_index, n_countries)
            country = np.where(np.isnan(country_new), country, country_new)
            world_new, _ = group_weighted_mean(country, country_rows, np.zeros(n_countries, dtype=int), 1)
            if np.all(np.isfinite(world_new[0])):
                world = world_new[0]
            broadcast = np.broadcast_to(world, (n_clients, world.size))
            round_objective = objective(problem, broadcast, broadcast, 0.0)
            history.append({"round": round_no, "participants": participants,
                            "world_objective": float(np.average(round_objective, weights=np.maximum(n, 1e-12)))})
        return world, country, district, history

    def fit_local_only(self, problem: GLMProblem, init: np.ndarray) -> np.ndarray:
        """Benchmark: each PHC trains alone on its own rows, no federation."""
        params = np.repeat(init[None].astype(float), problem.X.shape[0], axis=0)
        for _ in range(self.cfg.local_only_steps):
            params = newton_step(problem, params, anchor=params, mu_prox=0.0)
        return params


def initial_params(problem: GLMProblem) -> np.ndarray:
    """Intercept at the pooled mean (computed from per-client sums), all slopes zero."""
    total = problem.n.sum()
    mean_y = (problem.w * problem.y).sum() / max(total, 1.0)
    init = np.zeros(problem.X.shape[-1])
    init[0] = np.log(max(mean_y, 1e-6)) if problem.family == "poisson" else mean_y
    return init


def predict_rows(X: np.ndarray, params: np.ndarray, family: str) -> np.ndarray:
    return inverse_link(linear_predictor(X, params), family)
