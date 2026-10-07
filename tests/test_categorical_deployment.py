"""Preserve maintained-policy behavior at representable, almost-tied logits."""

import numpy as np
import pytest

pytest.importorskip("sb3_contrib")


@pytest.mark.parametrize("family", ["full_map", "partial", "recurrent"])
@pytest.mark.parametrize("default_dtype", ["float32", "float64"])
def test_probability_rounding_ties_match_selected_library_policy(tmp_path, family, default_dtype):
    import torch
    from stable_baselines3 import PPO

    from decision_lab.modern import ALGORITHMS, environment
    from decision_lab.modern_deployment import export
    from decision_lab.simulation_deployment import export_actor
    from decision_lab.simulation_envs import make_environment

    torch.set_num_threads(1)
    algorithm = "RecurrentPPO" if family == "recurrent" else "PPO"
    case = dict(
        env="MiniGrid-DoorKey-6x6-v0" if family == "full_map" else "MiniGrid-Empty-8x8-v0",
        algorithm=algorithm,
        profile="sparse" if family == "full_map" else "partial",
    )
    env = make_environment(case["env"]) if family == "full_map" else environment(case)
    parameters = dict(net_arch=[128, 128])
    if family == "recurrent":
        parameters.update(lstm_hidden_size=128, n_lstm_layers=1)
    library = PPO if family == "full_map" else ALGORITHMS[algorithm]
    model = library(
        "MlpLstmPolicy" if family == "recurrent" else "MlpPolicy",
        env,
        n_steps=8,
        batch_size=8,
        policy_kwargs=parameters,
        device="cpu",
        seed=7,
    )
    original_dtype = torch.get_default_dtype()
    try:
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.fill_(-10)
            model.policy.action_net.bias[0] = 0
            model.policy.action_net.bias[1] = 1e-9
        # Raw float32 logits differ; categorical normalization rounds their
        # probabilities to an exact tie, which the library resolves to action 0.
        assert model.policy.action_net.bias.argmax().item() == 1
        observation, _ = env.reset(seed=31)
        reference, _ = model.predict(observation, deterministic=True)
        assert int(reference) == 0
        torch.set_default_dtype(getattr(torch, default_dtype))
        destination = tmp_path / "actor.npz"
        actor = (
            export_actor(model, case, destination, "0" * 64)
            if family == "full_map"
            else export(model, case, destination, "0" * 64)
        )
        actual, _ = actor.predict(observation)
        np.testing.assert_array_equal(actual, reference)
        assert torch.get_default_dtype() == getattr(torch, default_dtype)
    finally:
        torch.set_default_dtype(original_dtype)
        env.close()

    if family == "full_map":
        # Geometry's integer ratios must also enter the actor at its declared
        # precision when a scientific caller changes the default dtype.
        from decision_lab.geometry import GeometryExtractor

        geometric_case = dict(
            env="MiniGrid-FourRooms-v0", algorithm="PPO", profile="potential_geometry"
        )
        geometric_env = make_environment(geometric_case["env"])
        model = PPO(
            "MlpPolicy",
            geometric_env,
            n_steps=8,
            batch_size=8,
            device="cpu",
            seed=7,
            policy_kwargs=dict(net_arch=[128, 128], features_extractor_class=GeometryExtractor),
        )
        try:
            observation, _ = geometric_env.reset(seed=31)
            reference, _ = model.predict(observation, deterministic=True)
            torch.set_default_dtype(getattr(torch, default_dtype))
            actor = export_actor(model, geometric_case, tmp_path / "geometry.npz", "0" * 64)
            actual, _ = actor.predict(observation)
            np.testing.assert_array_equal(actual, reference)
            assert torch.get_default_dtype() == getattr(torch, default_dtype)
        finally:
            torch.set_default_dtype(original_dtype)
            geometric_env.close()
