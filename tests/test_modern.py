"""Observation privacy, episodic memory and independent exported inference."""

import numpy as np
import pytest

pytest.importorskip("sb3_contrib")

from decision_lab.modern import ALGORITHMS, PartialObservation, environment, evaluate
from decision_lab.modern_deployment import export, load


def test_partial_view_ignores_unobserved_state():
    case = dict(env="MiniGrid-Empty-8x8-v0", algorithm="PPO", profile="partial")
    env = environment(case)
    obs, _ = env.reset(seed=7)
    raw = env.unwrapped.gen_obs()
    expected = env.observation(raw)
    env.unwrapped.step_count = 100
    env.unwrapped.agent_pos = (1, 1)
    np.testing.assert_array_equal(env.observation(raw), expected)
    assert obs.shape == (151,) and env.observation_space.contains(obs)
    assert isinstance(env, PartialObservation)
    env.close()


@pytest.mark.parametrize("algorithm", ["PPO", "RecurrentPPO", "TD3", "TQC"])
def test_portable_actor_matches_sequences_and_rejects_corruption(tmp_path, algorithm):
    import torch

    torch.set_num_threads(1)
    case = dict(
        env="MiniGrid-Empty-8x8-v0" if algorithm in {"PPO", "RecurrentPPO"} else "HalfCheetah-v5",
        algorithm=algorithm,
        profile="partial" if algorithm in {"PPO", "RecurrentPPO"} else "official",
    )
    env = environment(case)
    kwargs = dict(
        policy_kwargs=dict(net_arch=[128, 128] if case["profile"] == "partial" else [256, 256]),
        device="cpu",
        seed=3,
    )
    if algorithm == "RecurrentPPO":
        kwargs["policy_kwargs"].update(lstm_hidden_size=128, n_lstm_layers=1)
    if algorithm in {"PPO", "RecurrentPPO"}:
        kwargs.update(n_steps=8, batch_size=8)
    model = ALGORITHMS[algorithm](
        "MlpLstmPolicy" if algorithm == "RecurrentPPO" else "MlpPolicy", env, **kwargs
    )
    path = tmp_path / "actor.npz"
    actor = export(model, case, path, "0" * 64)
    memory, portable_memory = None, None
    obs, _ = env.reset(seed=31)
    for step in range(18):
        start = np.array([step in [0, 9]])
        reference, memory = model.predict(
            obs, state=memory, episode_start=start, deterministic=True
        )
        actual, portable_memory = actor.predict(obs, state=portable_memory, episode_start=start)
        np.testing.assert_allclose(actual, reference, rtol=0, atol=2e-7)
        if algorithm == "RecurrentPPO":
            np.testing.assert_allclose(portable_memory, memory, rtol=1e-6, atol=1e-7)
        obs, _, term, trunc, _ = env.step(reference)
        if term or trunc:
            obs, _ = env.reset(seed=32)
            memory, portable_memory = None, None
    first = evaluate(actor, case, [300000])
    repeated = evaluate(actor, case, [300000, 300001, 300000])
    assert first["returns"][0] == repeated["returns"][-1]
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        load(path)
    env.close()
