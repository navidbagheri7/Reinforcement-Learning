"""Finite MDP operators, planning and occupancy certificates.

P is a full stochastic transition kernel. continuation <= P excludes true terminal
transitions from bootstrap; time limits do not change this operator.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class FiniteMDP:
    transition: np.ndarray
    reward: np.ndarray
    legal: np.ndarray | None = None
    continuation: np.ndarray | None = None

    def __post_init__(self):
        self.transition = np.array(self.transition, dtype=float, copy=True)
        self.reward = np.array(self.reward, dtype=float, copy=True)
        if self.transition.ndim != 3:
            raise ValueError("Expected S by A by S transition probabilities")
        s, a, last = self.transition.shape
        if s != last or self.reward.shape != (s, a):
            raise ValueError("Incompatible state/action dimensions")
        self.legal = (
            np.ones((s, a), dtype=bool)
            if self.legal is None
            else np.asarray(self.legal, dtype=bool)
        )
        self.continuation = (
            self.transition.copy()
            if self.continuation is None
            else np.asarray(self.continuation, dtype=float)
        )
        if self.legal.shape != (s, a) or not self.legal.any(1).all():
            raise ValueError("Every state needs a legal action")
        if (
            not np.isfinite(self.transition).all()
            or not np.isfinite(self.reward).all()
            or np.any(self.transition < 0)
            or not np.allclose(self.transition.sum(2), 1, atol=1e-12)
        ):
            raise ValueError("Finite rewards and stochastic transition rows required")
        if (
            self.continuation.shape != self.transition.shape
            or not np.isfinite(self.continuation).all()
            or np.any(self.continuation < 0)
            or np.any(self.continuation > self.transition + 1e-12)
        ):
            raise ValueError("Continuation probabilities must lie between zero and P")

    @property
    def states(self):
        return self.reward.shape[0]

    @property
    def actions(self):
        return self.reward.shape[1]

    def q(self, value, gamma):
        discount(gamma)
        value = np.asarray(value, dtype=float)
        if value.shape != (self.states,) or not np.isfinite(value).all():
            raise ValueError("Finite state value vector required")
        q = self.reward + gamma * (self.continuation.reshape(-1, self.states) @ value).reshape(
            self.reward.shape
        )
        return np.where(self.legal, q, -np.inf)

    def policy_matrix(self, policy):
        p = np.asarray(policy)
        if p.shape == (self.states,):
            if not np.issubdtype(p.dtype, np.integer) or np.any((p < 0) | (p >= self.actions)):
                raise ValueError("Policy action IDs must be integers in range")
            p = np.eye(self.actions)[p]
        if (
            p.shape != self.reward.shape
            or not np.isfinite(p).all()
            or np.any(p < 0)
            or np.any(p[~self.legal] > 0)
            or not np.allclose(p.sum(1), 1, atol=1e-12)
        ):
            raise ValueError("Policy must distribute probability over legal actions")
        return p


def discount(gamma):
    if not np.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("Discounted planning requires 0 <= gamma < 1")


def evaluate_policy(mdp, policy, gamma=0.99):
    discount(gamma)
    policy = mdp.policy_matrix(policy)
    kernel = np.einsum("sa,sat->st", policy, mdp.continuation)
    reward = (policy * mdp.reward).sum(1)
    return np.linalg.solve(np.eye(mdp.states) - gamma * kernel, reward)


def value_iteration(mdp, gamma=0.99, tol=1e-10, max_iter=10000):
    value = np.zeros(mdp.states)
    history = []
    for step in range(max_iter):
        new = mdp.q(value, gamma).max(1)
        residual = float(np.max(abs(new - value)))
        history.append(residual)
        value = new
        if residual <= tol:
            break
    residual = float(np.max(abs(mdp.q(value, gamma).max(1) - value)))
    return {
        "value": value,
        "policy": mdp.q(value, gamma).argmax(1),
        "residual": residual,
        "error_bound": residual / (1 - gamma),
        "iterations": step + 1,
        "history": np.array(history),
        "converged": residual <= tol,
    }


def policy_iteration(mdp, gamma=0.99, max_iter=1000):
    policy = mdp.legal.argmax(1)
    history = []
    for step in range(max_iter):
        value = evaluate_policy(mdp, policy, gamma)
        q = mdp.q(value, gamma)
        new = q.argmax(1)
        residual = float(np.max(abs(q.max(1) - value)))
        history.append(residual)
        # Preserve a tied incumbent to prevent numerical policy cycling.
        improvement = q[np.arange(mdp.states), new] - q[np.arange(mdp.states), policy]
        new = np.where(improvement > 1e-10, new, policy)
        if np.array_equal(new, policy):
            break
        policy = new
    return {
        "value": evaluate_policy(mdp, policy, gamma),
        "policy": policy,
        "iterations": step + 1,
        "history": np.array(history),
    }


def finite_horizon(mdp, horizon, gamma=1.0):
    if not isinstance(horizon, int) or horizon < 1 or not 0 <= gamma <= 1:
        raise ValueError("Positive horizon and discount in [0,1] required")
    value = np.zeros((horizon + 1, mdp.states))
    policy = np.empty((horizon, mdp.states), int)
    kernel = mdp.continuation.reshape(-1, mdp.states)
    for t in reversed(range(horizon)):
        q = mdp.reward + gamma * (kernel @ value[t + 1]).reshape(mdp.reward.shape)
        q = np.where(mdp.legal, q, -np.inf)
        value[t] = q.max(1)
        policy[t] = q.argmax(1)
    return value, policy


def initial_distribution(mdp, mu):
    mu = np.ones(mdp.states) / mdp.states if mu is None else np.asarray(mu, float)
    if (
        mu.shape != (mdp.states,)
        or np.any(mu < 0)
        or not np.isfinite(mu).all()
        or not np.isclose(mu.sum(), 1)
    ):
        raise ValueError("Initial distribution must be a probability vector")
    return mu


def discounted_occupancy(mdp, gamma=0.99, mu=None, costs=None, budgets=None):
    """Maximize reward, with normalized discounted cost upper bounds."""
    discount(gamma)
    mu = initial_distribution(mdp, mu)
    s, a = np.nonzero(mdp.legal)
    p = mdp.continuation[s, a]
    flow = np.eye(mdp.states)[:, s] - gamma * p.T
    ub = rhs = None
    if costs is not None:
        costs = np.asarray(costs, float)
        budgets = np.asarray(budgets, float)
        if (
            costs.ndim != 3
            or costs.shape[1:] != mdp.reward.shape
            or budgets.shape != (len(costs),)
            or not np.isfinite(costs).all()
            or not np.isfinite(budgets).all()
        ):
            raise ValueError("Costs require C by S by A arrays and C finite budgets")
        ub = (1 - gamma) * costs[:, s, a]
        rhs = budgets
    result = linprog(
        -mdp.reward[s, a], A_eq=flow, b_eq=mu, A_ub=ub, b_ub=rhs, bounds=(0, None), method="highs"
    )
    if not result.success:
        raise RuntimeError(f"Occupancy problem: {result.message}")
    x = np.zeros(mdp.reward.shape)
    x[s, a] = result.x
    mass = x.sum(1)
    policy = np.eye(mdp.actions)[mdp.legal.argmax(1)]
    np.divide(x, mass[:, None], out=policy, where=mass[:, None] > 1e-14)
    value = evaluate_policy(mdp, policy, gamma)
    return {
        "occupancy": x,
        "policy": policy,
        "value": value,
        "objective": float(mu @ value),
        "flow_residual": float(np.max(abs(flow @ result.x - mu))),
        "normalized_costs": None if costs is None else (1 - gamma) * (costs * x).sum((1, 2)),
        "budget_shadow_prices": None if costs is None else -result.ineqlin.marginals,
        "dual_objective": float(
            mu @ result.eqlin.marginals + (0 if rhs is None else rhs @ result.ineqlin.marginals)
        )
        * -1,
    }


def bellman_lp(mdp, gamma=0.99, mu=None):
    discount(gamma)
    mu = initial_distribution(mdp, mu)
    s, a = np.nonzero(mdp.legal)
    matrix = gamma * mdp.continuation[s, a] - np.eye(mdp.states)[s]
    result = linprog(
        mu, A_ub=matrix, b_ub=-mdp.reward[s, a], bounds=[(None, None)] * mdp.states, method="highs"
    )
    if not result.success:
        raise RuntimeError(result.message)
    return {
        "value": result.x,
        "objective": float(result.fun),
        "residual": float(np.max(abs(mdp.q(result.x, gamma).max(1) - result.x))),
    }


def average_occupancy(mdp):
    """Stationary occupancy optimum; initial-state optimality needs recurrence scope."""
    s, a = np.nonzero(mdp.legal)
    flow = np.eye(mdp.states)[:, s] - mdp.transition[s, a].T
    eq = np.vstack([flow, np.ones(len(s))])
    rhs = np.r_[np.zeros(mdp.states), 1.0]
    result = linprog(-mdp.reward[s, a], A_eq=eq, b_eq=rhs, bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    x = np.zeros(mdp.reward.shape)
    x[s, a] = result.x
    mass = x.sum(1)
    policy = np.eye(mdp.actions)[mdp.legal.argmax(1)]
    np.divide(x, mass[:, None], out=policy, where=mass[:, None] > 1e-14)
    matrix = np.column_stack([-np.ones(len(s)), mdp.transition[s, a] - np.eye(mdp.states)[s]])
    bounds = [(None, None)] * (mdp.states + 1)
    bounds[1] = (0, 0)
    dual = linprog(
        np.r_[1.0, np.zeros(mdp.states)],
        A_ub=matrix,
        b_ub=-mdp.reward[s, a],
        bounds=bounds,
        method="highs",
    )
    if not dual.success:
        raise RuntimeError(dual.message)
    return {
        "occupancy": x,
        "policy": policy,
        "average_reward": float(-result.fun),
        "gain_dual": float(dual.fun),
        "bias": dual.x[1:],
        "stationarity_residual": float(np.max(abs(flow @ result.x))),
    }
