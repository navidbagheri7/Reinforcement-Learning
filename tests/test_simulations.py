"""Check state visibility, seed pairing, and potential/reward accounting."""

import numpy as np
import pytest

pytest.importorskip("minigrid")
from decision_lab.simulation_envs import make_environment


@pytest.mark.parametrize("name", ["MiniGrid-FourRooms-v0", "MiniGrid-DoorKey-6x6-v0"])
def test_seeded_observations_and_shaping_telescope(name):
    sparse = make_environment(name)
    shaped = make_environment(name, shaping=True)
    try:
        first, _ = sparse.reset(seed=51)
        second, _ = shaped.reset(seed=51)
        assert np.array_equal(first, second)
        assert sparse.observation_space.contains(first)
        initial = shaped.previous_potential
        accumulated = 0
        rng = np.random.default_rng(10)
        for t in range(30):
            action = int(rng.integers(7))
            a, r, term, trunc, info = sparse.step(action)
            b, sr, ster, stru, sinfo = shaped.step(action)
            assert np.array_equal(a, b)
            assert (term, trunc) == (ster, stru)
            assert sinfo["official_reward"] == r
            assert sr == pytest.approx(r + sinfo["shaping_reward"])
            accumulated += 0.99**t * (sr - r)
            assert sparse.observation_space.contains(a)
            if term or trunc:
                break
        expected = -initial + 0.99 ** (t + 1) * shaped.previous_potential
        assert accumulated == pytest.approx(expected)
    finally:
        sparse.close()
        shaped.close()


def test_time_limit_retains_bootstrap_potential():
    env = make_environment("MiniGrid-DoorKey-6x6-v0", shaping=True)
    try:
        env.reset(seed=3)
        env.unwrapped.max_steps = 1
        _, _, terminated, truncated, info = env.step(0)
        assert truncated and not terminated
        assert env.previous_potential < 0
        assert "official_reward" in info
    finally:
        env.close()


@pytest.mark.parametrize("name", ["MiniGrid-FourRooms-v0", "MiniGrid-DoorKey-6x6-v0"])
def test_known_model_planner_reaches_goal_with_reward_identity(name):
    from decision_lab.navigation_planner import Planner

    env = make_environment(name, shaping=True)
    try:
        for seed in [0, 51, 300000]:
            observation, _ = env.reset(seed=seed)
            planner = Planner(env)
            total = 0.0
            while True:
                action, _ = planner.predict(observation)
                observation, reward, term, trunc, info = env.step(action)
                total += info["official_reward"]
                if term or trunc:
                    break
            assert term and not trunc and total > 0
            assert total == pytest.approx(
                1 - 0.9 * env.unwrapped.step_count / env.unwrapped.max_steps
            )
            assert env.previous_potential == 0
    finally:
        env.close()


def test_public_actor_roundtrip_matches_library_action_and_rejects_changes(tmp_path):
    pytest.importorskip("stable_baselines3")
    import torch
    from stable_baselines3 import PPO, SAC

    from decision_lab.simulation_deployment import export_actor, load_actor

    torch.set_num_threads(1)
    for name, algorithm, profile in [
        ("MiniGrid-DoorKey-6x6-v0", PPO, "sparse"),
        ("Pendulum-v1", SAC, "official"),
        ("Pendulum-v1", PPO, "official"),
    ]:
        env = make_environment(name)
        try:
            model = algorithm(
                "MlpPolicy", env, policy_kwargs=dict(net_arch=[16, 16]), device="cpu", seed=2
            )
            path = tmp_path / f"{name}.npz"
            actor = export_actor(
                model, dict(env=name, algorithm=algorithm.__name__, profile=profile), path, "0" * 64
            )
            obs, _ = env.reset(seed=42)
            np.testing.assert_array_equal(
                actor.predict(obs)[0], model.predict(obs, deterministic=True)[0]
            )
            if name == "Pendulum-v1":
                assert (
                    actor.predict(obs)[0].dtype
                    == model.predict(obs, deterministic=True)[0].dtype
                    == np.float32
                )
            path.write_bytes(path.read_bytes() + b"changed")
            with pytest.raises(ValueError, match="changed"):
                load_actor(path)
        finally:
            env.close()


def test_geometry_encodes_local_walls_and_requires_declared_shape():
    torch = pytest.importorskip("torch")
    from decision_lab.navigation_features import fourrooms_features

    env = make_environment("MiniGrid-FourRooms-v0")
    try:
        obs, _ = env.reset(seed=23)
        features = fourrooms_features(torch.tensor(obs[None]))
        assert features.shape == (1, 44)
        np.testing.assert_array_equal(features.numpy()[0, :19], obs[-19:])
        x, y = env.unwrapped.agent_pos
        expected = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = env.unwrapped.grid.get(x + dx, y + dy)
                expected.append(float(cell is not None and cell.type == "wall"))
        np.testing.assert_array_equal(features.numpy()[0, -9:], expected)
        with pytest.raises(ValueError, match="19x19"):
            fourrooms_features(torch.zeros(1, 127))
    finally:
        env.close()


def test_curriculum_changes_only_training_starts_and_finishes_at_official_distribution():
    from decision_lab.curriculum import StartCurriculum

    curriculum = StartCurriculum(
        make_environment("MiniGrid-FourRooms-v0", shaping=True),
        [dict(until_fraction=0.75, max_distance=3)],
        8,
        1000,
    )
    official = make_environment("MiniGrid-FourRooms-v0", shaping=True)
    try:
        obs, info = curriculum.reset(seed=42)
        pos = tuple(curriculum.unwrapped.agent_pos)
        assert 0 < curriculum.env.goal_field[pos] <= 3
        assert info["curriculum_distance_cap"] == 3
        assert curriculum.env.previous_potential == curriculum.env.potential()
        curriculum.interactions = 125
        a, info = curriculum.reset(seed=42)
        b, _ = official.reset(seed=42)
        assert info["curriculum_distance_cap"] is None
        np.testing.assert_array_equal(a, b)
    finally:
        curriculum.close()
        official.close()


def test_geometry_actor_roundtrip_has_identical_actions(tmp_path):
    from stable_baselines3 import PPO

    from decision_lab.geometry import GeometryExtractor
    from decision_lab.simulation_deployment import export_actor

    env = make_environment("MiniGrid-FourRooms-v0")
    try:
        model = PPO(
            "MlpPolicy",
            env,
            seed=2,
            device="cpu",
            policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=GeometryExtractor),
        )
        actor = export_actor(
            model,
            dict(env="MiniGrid-FourRooms-v0", algorithm="PPO", profile="potential_geometry"),
            tmp_path / "actor.npz",
            "0" * 64,
        )
        assert actor.metadata["encoder"] == "fourrooms_geometry_v1"
        assert actor.metadata["dimensions"][0] == 44
        for seed in [0, 47, 300000]:
            obs, _ = env.reset(seed=seed)
            np.testing.assert_array_equal(
                actor.predict(obs)[0], model.predict(obs, deterministic=True)[0]
            )
    finally:
        env.close()


def test_custom_full_output_cannot_overwrite_published_evidence(tmp_path, monkeypatch):
    import json
    import sys

    from decision_lab.cli import main

    monkeypatch.chdir(tmp_path)
    evidence = tmp_path / "assets/source_data/simulations/Example_PPO_sparse_0.npz"
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(b"published evidence must remain unchanged")
    config = tmp_path / "protocol.json"
    config.write_text(
        json.dumps(
            dict(device="cpu", cases=[dict(env="Example", algorithm="PPO", profile="sparse")])
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["decision-lab", "simulations", "--config", str(config), "--output", "results/custom"],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert evidence.read_bytes() == b"published evidence must remain unchanged"
