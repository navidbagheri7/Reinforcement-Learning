import numpy as np
import pytest

from decision_lab.wireless import WirelessEnv, WirelessModel


@pytest.mark.parametrize("profile,size", [("per_user", 324), ("global", 108)])
def test_physics_kernel_and_accounting(profile, size):
    model = WirelessModel(profile=profile)
    mdp = model.finite_mdp()
    assert mdp.states == size
    np.testing.assert_allclose(mdp.transition.sum(2), 1, atol=1e-12)
    # Full battery saturation must retain every energy-arrival outcome's mass.
    state = model.index[(1, 2, 2, 1, 2, 2) if profile == "per_user" else (1, 2, 1, 2, 2)]
    expected = np.sum(mdp.transition[state, 0] * model.states[:, 1])
    assert abs(expected - 2.0) < 1e-12
    rng = np.random.default_rng(37)
    for _ in range(100):
        state = int(rng.integers(size))
        action = int(rng.integers(4))
        nxt, reward, info = model.outcome(state, action, *model.draw(rng))
        assert mdp.transition[state, action, nxt] > 0
        assert reward == mdp.reward[state, action]
        assert info["served"] + info["expired"] == info["offered"]
        assert not info["invalid_action"] or info["served"] == 0


def test_joint_channel_eligibility_and_one_slot_expiry():
    model = WirelessModel()
    state = model.index[(1, 1, 2, 1, 1, 1)]
    assert model.legal[state, 1] and model.legal[state, 2] and not model.legal[state, 3]
    nxt, r, info = model.outcome(state, 0, np.zeros(2, int), np.zeros(2, int), np.zeros(2, int))
    np.testing.assert_array_equal(model.unpack(nxt)[0], [0, 0])
    assert r == -2 and info["expired"] == 2
    nxt, r, info = model.outcome(state, 1, np.zeros(2, int), np.ones(2, int), np.zeros(2, int))
    assert info["served"] == 1 and info["expired"] == 1 and r == -1
    np.testing.assert_array_equal(model.unpack(nxt)[1], [1, 2])


def test_simulator_transition_frequencies_match_independent_bernoulli_moments():
    model = WirelessModel(p_data=0.7, p_energy=0.3)
    env = WirelessEnv(model, horizon=1)
    state = model.index[(0, 0, 0, 0, 0, 0)]
    values = []
    for seed in range(5000):
        env.reset(seed=seed, options={"state": state})
        obs, _, terminated, truncated, info = env.step(0)
        values.append(model.states[info["state"]])
        assert not terminated and truncated and env.observation_space.contains(obs)
    average = np.mean(values, 0)
    np.testing.assert_allclose(average, [0.7, 0.3, 1, 0.7, 0.3, 1], atol=0.035)
