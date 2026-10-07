import numpy as np
import pytest

from decision_lab.mdp import (
    FiniteMDP,
    average_occupancy,
    bellman_lp,
    discounted_occupancy,
    evaluate_policy,
    finite_horizon,
    policy_iteration,
    value_iteration,
)


def test_geometric_value_and_primal_dual_certificate():
    m = FiniteMDP(np.ones((1, 2, 1)), np.array([[1.0, 2.0]]))
    vi = value_iteration(m, 0.9)
    np.testing.assert_allclose(vi["value"], [20.0], atol=1e-8)
    np.testing.assert_array_equal(vi["policy"], [1])
    assert vi["error_bound"] >= abs(vi["value"][0] - 20.0) - 1e-12
    np.testing.assert_allclose(policy_iteration(m, 0.9)["value"], [20.0])
    primal = bellman_lp(m, 0.9)
    dual = discounted_occupancy(m, 0.9)
    assert abs(primal["objective"] - dual["objective"]) < 1e-10
    assert abs(dual["objective"] - dual["dual_objective"]) < 1e-10
    assert dual["flow_residual"] < 1e-12


def test_constrained_policy_randomizes_and_shadow_price_matches_perturbation():
    m = FiniteMDP(np.ones((1, 2, 1)), np.array([[0.0, 2.0]]))
    costs = np.array([[[0.0, 1.0]]])
    result = discounted_occupancy(m, 0.9, costs=costs, budgets=[0.3])
    np.testing.assert_allclose(result["policy"], [[0.7, 0.3]], atol=1e-10)
    assert abs(result["normalized_costs"][0] - 0.3) < 1e-10
    changed = discounted_occupancy(m, 0.9, costs=costs, budgets=[0.3001])
    np.testing.assert_allclose(
        (changed["objective"] - result["objective"]) / 0.0001,
        result["budget_shadow_prices"][0],
        rtol=1e-7,
    )


def test_terminal_transitions_stop_bootstrap_and_masks_exclude_fake_reward():
    p = np.ones((1, 2, 1))
    m = FiniteMDP(p, [[3.0, 999.0]], legal=[[True, False]], continuation=np.zeros_like(p))
    np.testing.assert_allclose(evaluate_policy(m, [0], 0.99), [3.0])
    np.testing.assert_allclose(value_iteration(m, 0.99)["value"], [3.0])
    values, actions = finite_horizon(m, 5, 1.0)
    np.testing.assert_allclose(values[:-1], 3.0)
    np.testing.assert_array_equal(actions, 0)
    with pytest.raises(ValueError):
        evaluate_policy(m, [1], 0.9)


def test_average_reward_two_state_chain_matches_cycle_arithmetic():
    p = np.array([[[0.0, 1.0]], [[1.0, 0.0]]])
    m = FiniteMDP(p, [[2.0], [4.0]])
    result = average_occupancy(m)
    assert abs(result["average_reward"] - 3.0) < 1e-12
    assert abs(result["gain_dual"] - 3.0) < 1e-12
    np.testing.assert_allclose(result["occupancy"], 0.5)
    assert result["stationarity_residual"] < 1e-12


def test_malformed_kernel_and_policy_are_rejected():
    with pytest.raises(ValueError):
        FiniteMDP([[[0.9]]], [[1.0]])
    with pytest.raises(ValueError):
        FiniteMDP([[[1.0]]], [[np.inf]])
    m = FiniteMDP([[[1.0]]], [[1.0]])
    with pytest.raises(ValueError):
        value_iteration(m, 1.0)
    with pytest.raises(ValueError):
        evaluate_policy(m, [[0.8]])
