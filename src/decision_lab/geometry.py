"""Controlled PPO comparison using explicit full-map geometry features."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .artifacts import Run, interval, sha256
from .navigation_features import fourrooms_features
from .simulation_envs import make_environment
from .simulations import ALGORITHMS, Selection, evaluate, identifier


class GeometryExtractor(BaseFeaturesExtractor):
    encoder = "fourrooms_geometry_v1"

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=44)
        if observation_space.shape != (1102,):
            raise ValueError("This geometry extractor supports Four Rooms only")

    def forward(self, observation):
        return fourrooms_features(observation)


def train_case(config, case, seed, output_root="results/reference/geometry", quick=False):
    if (case["env"], case["algorithm"], case["profile"]) != (
        "MiniGrid-FourRooms-v0",
        "PPO",
        "potential_geometry",
    ):
        raise ValueError("Geometry protocol supports the declared Four Rooms PPO preset only")
    directory = Path(output_root) / identifier(case) / str(seed)
    complete_config = dict(benchmark=config, case=case, seed=seed, quick=quick)
    if (directory / "run.json").exists():
        previous = json.loads((directory / "run.json").read_text())
        if previous["status"] == "complete" and previous["config"] == complete_config:
            print(f"Already complete: {directory}", flush=True)
            return
        raise ValueError(
            f"Existing incomplete or changed protocol: {directory}; choose a new output"
        )
    run = Run(
        directory,
        complete_config,
        [
            Path(__file__),
            Path(__file__).with_name("navigation_features.py"),
            Path(__file__).with_name("simulations.py"),
            Path(__file__).with_name("simulation_envs.py"),
            Path(__file__).with_name("artifacts.py"),
        ],
    )
    run.record["versions"].update({p: version(p) for p in ["minigrid", "mujoco"]})
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir()
    torch.set_num_threads(1)
    env = make_vec_env(
        lambda: make_environment(case["env"], shaping=case["profile"].startswith("potential")),
        n_envs=case["n_envs"],
        seed=seed,
        monitor_dir=str(checkpoints / "monitor"),
    )
    parameters = dict(
        learning_rate=3e-4,
        seed=seed,
        device=config["device"],
        policy_kwargs=dict(net_arch=[256, 256], features_extractor_class=GeometryExtractor),
        gamma=0.99,
        verbose=0,
    )
    if case["algorithm"] == "PPO":
        parameters.update(
            n_steps=256,
            batch_size=256,
            n_epochs=10,
            ent_coef=0.01 if case["env"].startswith("MiniGrid") else 0.0,
            gae_lambda=0.95,
        )
    else:
        parameters.update(
            buffer_size=1000000,
            learning_starts=10000 if not quick else 128,
            batch_size=256,
            train_freq=4,
            gradient_steps=1,
            ent_coef="auto",
        )
    model = ALGORITHMS[case["algorithm"]]("MlpPolicy", env, **parameters)
    callback = Selection(case, config, checkpoints)
    try:
        model.learn(case["steps"], callback=callback)
        callback.check()
        model.save(checkpoints / "final_model")
        selected = ALGORITHMS[case["algorithm"]].load(checkpoints / "best_model", device="cpu")
        test = evaluate(selected, case, np.arange(config["test_episodes"]) + 300000)
    finally:
        env.close()
    public = Path("assets/source_data/simulations") if not quick else directory
    public.mkdir(parents=True, exist_ok=True)
    numeric = public / f"{identifier(case)}_{seed}.npz"
    np.savez_compressed(numeric, **test)
    metrics = dict(
        env=case["env"],
        algorithm=case["algorithm"],
        profile=case["profile"],
        seed=seed,
        actual_steps=model.num_timesteps,
        selected_step=callback.best_step,
        best_validation_return=callback.best,
        test=interval(test["returns"], seed),
        mean_length=float(test["lengths"].mean()),
        termination_fraction=float(test["terminated"].mean()),
        truncation_fraction=float(test["truncated"].mean()),
        success_fraction=float((test["returns"] > 0).mean())
        if case["env"].startswith("MiniGrid")
        else None,
        validation_history=callback.history,
        checkpoint_sha256=sha256(checkpoints / "best_model.zip"),
        numeric_sha256=sha256(numeric),
        observation_profile="global symbolic map and engineered pose/object features"
        if case["env"].startswith("MiniGrid")
        else "official proprioceptive state",
        evaluation_reward="official unmodified reward; shaping disabled",
        selection="best mean official return on validation seeds 200000..200011; first tie",
        test_seeds="300000..300099; independent of fitting seed",
    )
    run.finish(metrics)
    print(f"COMPLETE {identifier(case)} seed={seed} test={metrics['test']['mean']:.4f}", flush=True)
