"""Batched GLM primitives for simulated federated clients.

Arrays carry a leading client axis C. Every per-client quantity is a sum over that client's own
rows, so batching is a simulation speed-up, not pooling.

Families: Poisson with log link (counts, multiplicative outbreak effects) and Gaussian with
identity link (bounded percentages).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

MAX_ETA = 20.0
JITTER = 1e-6


@dataclass
class GLMProblem:
    X: np.ndarray  # (C, N, F) standardized design, intercept in column 0
    y: np.ndarray  # (C, N) target, zero where w == 0
    w: np.ndarray  # (C, N) row weights (0/1)
    family: str
    l2: float

    @classmethod
    def build(cls, Z: np.ndarray, y: np.ndarray, mask: np.ndarray, family: str, l2: float) -> GLMProblem:
        return cls(X=Z, y=np.where(mask, np.nan_to_num(y), 0.0), w=mask.astype(float), family=family, l2=l2)

    @cached_property
    def n(self) -> np.ndarray:
        return self.w.sum(axis=1)

    @cached_property
    def penalty_mask(self) -> np.ndarray:
        mask = np.ones(self.X.shape[-1])
        mask[0] = 0.0  # never shrink the intercept
        return mask


def linear_predictor(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    return (X @ P[..., None])[..., 0]


def inverse_link(eta: np.ndarray, family: str) -> np.ndarray:
    return np.exp(np.clip(eta, -MAX_ETA, MAX_ETA)) if family == "poisson" else eta


def _terms(problem: GLMProblem, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-level loss, mean and curvature at per-client parameters P (C, F)."""
    eta = linear_predictor(problem.X, P)
    if problem.family == "poisson":
        # Overflow guard for the log link only; identity-link targets (e.g. bed occupancy ~60) must not be clipped.
        eta = np.clip(eta, -MAX_ETA, MAX_ETA)
        mu = np.exp(eta)
        return problem.w * (mu - problem.y * eta), mu, problem.w * mu
    return 0.5 * problem.w * (eta - problem.y) ** 2, eta, problem.w


def client_loss(problem: GLMProblem, P: np.ndarray) -> np.ndarray:
    """(C,) summed negative log-likelihood of each client's rows."""
    return _terms(problem, P)[0].sum(axis=1)


def client_statistics(problem: GLMProblem, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summed loss (C,), gradient (C, F) and Hessian (C, F, F): the only messages a client sends."""
    loss, mu, curvature = _terms(problem, P)
    Xt = problem.X.transpose(0, 2, 1)
    grad = (Xt @ (problem.w * (mu - problem.y))[..., None])[..., 0]
    hess = Xt @ (problem.X * curvature[..., None])
    return loss.sum(axis=1), grad, hess


def _penalty(P: np.ndarray, anchor: np.ndarray, l2: float, mu, penalty_mask: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    return 0.5 * l2 * ((P * penalty_mask) ** 2).sum(axis=1) + 0.5 * mu * ((P - anchor) ** 2).sum(axis=1)


def objective(problem: GLMProblem, P: np.ndarray, anchor: np.ndarray, mu_prox) -> np.ndarray:
    """(C,) mean negative log-likelihood per client plus L2 and proximal penalties."""
    data = client_loss(problem, P) / np.maximum(problem.n, 1.0)
    return data + _penalty(P, anchor, problem.l2, mu_prox, problem.penalty_mask)


def _damped_solve(hess: np.ndarray, grad: np.ndarray, diagonal: np.ndarray) -> np.ndarray:
    hess = hess.copy()
    idx = np.arange(hess.shape[-1])
    hess[:, idx, idx] += diagonal
    return np.linalg.solve(hess, grad[..., None])[..., 0]


def newton_step(problem: GLMProblem, P: np.ndarray, anchor: np.ndarray, mu_prox: float,
                max_backtracks: int = 12) -> np.ndarray:
    """One damped Newton step per client on its own objective (FedAvg local update, local-only fits)."""
    n = np.maximum(problem.n, 1.0)
    _, grad, hess = client_statistics(problem, P)
    grad = grad / n[:, None] + problem.l2 * problem.penalty_mask * P + mu_prox * (P - anchor)
    direction = _damped_solve(hess / n[:, None, None], grad,
                              problem.l2 * problem.penalty_mask + mu_prox + JITTER)
    base = objective(problem, P, anchor, mu_prox)
    step = np.ones(P.shape[0])
    candidate = P - direction
    worse = ~(objective(problem, candidate, anchor, mu_prox) <= base + 1e-12)
    for _ in range(max_backtracks):
        if not worse.any():
            break
        step = np.where(worse, step * 0.5, step)
        candidate = P - step[:, None] * direction
        worse = ~(objective(problem, candidate, anchor, mu_prox) <= base + 1e-12)
    return np.where(worse[:, None], P, candidate)


def _group_sum(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    if n_groups == len(groups) and np.array_equal(groups, np.arange(n_groups)):
        return values.copy()
    out = np.zeros((n_groups,) + values.shape[1:])
    np.add.at(out, groups, values)
    return out


def group_newton(problem: GLMProblem, start: np.ndarray, groups: np.ndarray, n_groups: int,
                 prior_rows: float, steps: int, max_backtracks: int = 12) -> tuple[np.ndarray, list[dict]]:
    """Exact federated Newton for every group of clients (a district, a country, the world, or one PHC).

    Members evaluate loss/gradient/Hessian sums at their group's current model; the group adds them
    up and takes a penalized Newton step, so the result equals fitting the group's pooled rows.
    `prior_rows` shrinks each group toward `start` (its parent model) with the weight of that many
    rows, so shrinkage fades as the group has more data.
    """
    P = start.astype(float).copy()
    anchor = P.copy()
    n_features = P.shape[1]
    rows = np.bincount(groups, weights=problem.n, minlength=n_groups)
    N = np.maximum(rows, 1.0)
    active = rows > 0
    mask, l2 = problem.penalty_mask, problem.l2
    mu = np.zeros(n_groups)
    history: list[dict] = []

    def group_objective(params: np.ndarray) -> np.ndarray:
        loss = np.bincount(groups, weights=client_loss(problem, params[groups]), minlength=n_groups)
        return loss / N + _penalty(params, anchor, l2, mu, mask)

    for step in range(steps):
        loss, grad, hess = client_statistics(problem, P[groups])
        H = _group_sum(hess, groups, n_groups)
        if step == 0 and prior_rows > 0:
            curvature = np.trace(H, axis1=1, axis2=2) / (n_features * N)
            mu = prior_rows / N * curvature
        g = _group_sum(grad, groups, n_groups) / N[:, None] + l2 * mask * P + mu[:, None] * (P - anchor)
        diagonal = l2 * mask[None, :] + mu[:, None] + JITTER
        direction = _damped_solve(H / N[:, None, None], g, diagonal)

        base = np.bincount(groups, weights=loss, minlength=n_groups) / N + _penalty(P, anchor, l2, mu, mask)
        step_size = np.ones(n_groups)
        candidate = P - direction
        value = group_objective(candidate)
        worse = ~(value <= base + 1e-12)
        for _ in range(max_backtracks):
            if not (worse & active).any():
                break
            step_size = np.where(worse, step_size * 0.5, step_size)
            candidate = P - step_size[:, None] * direction
            value = group_objective(candidate)
            worse = ~(value <= base + 1e-12)
        keep = worse | ~active
        P = np.where(keep[:, None], P, candidate)
        accepted = np.where(keep, base, value)
        history.append({"step": step + 1, "objective": float(np.average(accepted, weights=np.maximum(rows, 1e-12)))})
    return P, history


def poisson_deviance(y: np.ndarray, mu: np.ndarray, w: np.ndarray) -> float:
    mu = np.maximum(mu, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    return float(2.0 * (w * term).sum() / max(w.sum(), 1.0))
