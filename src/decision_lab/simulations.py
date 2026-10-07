"""Train visual navigation and locomotion benchmarks with validation-only selection."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from .artifacts import Run, interval, sha256, write_json
from .simulation_envs import make_environment

ALGORITHMS = {"PPO": PPO, "SAC": SAC}


def identifier(case):
    return f"{case['env']}_{case['algorithm']}_{case['profile']}"


def evaluate(model, case, seeds, render=False):
    env = make_environment(case["env"], render_mode="rgb_array" if render else None)
    returns, lengths, terminateds, truncateds = [], [], [], []
    frames = []
    first_observations, first_actions = [], []
    try:
        for i, seed in enumerate(seeds):
            obs, _ = env.reset(seed=int(seed))
            total, length = 0.0, 0
            while True:
                if render and i == 0 and (case["env"].startswith("MiniGrid") or length % 4 == 0):
                    frames.append(env.render())
                action, _ = model.predict(obs, deterministic=True)
                if i == 0 and length < 32:
                    first_observations.append(obs.copy())
                    first_actions.append(np.asarray(action).copy())
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                length += 1
                if terminated or truncated:
                    if render and i == 0:
                        frames.append(env.render())
                    break
            returns.append(total)
            lengths.append(length)
            terminateds.append(terminated)
            truncateds.append(truncated)
    finally:
        env.close()
    result = dict(
        returns=np.array(returns),
        lengths=np.array(lengths),
        terminated=np.array(terminateds),
        truncated=np.array(truncateds),
        seeds=np.array(seeds),
        first_observations=np.array(first_observations),
        first_actions=np.array(first_actions),
    )
    return (result, frames) if render else result


class Selection(BaseCallback):
    def __init__(self, case, config, directory):
        super().__init__()
        self.case, self.config, self.directory = case, config, directory
        self.best = -np.inf
        self.best_step = None
        self.last = 0
        self.history = []

    def check(self):
        result = evaluate(
            self.model, self.case, np.arange(self.config["validation_episodes"]) + 200000
        )
        score = float(result["returns"].mean())
        self.last = self.num_timesteps
        self.history.append(
            dict(
                step=self.last,
                mean_return=score,
                returns=result["returns"].tolist(),
                updates=self.model._n_updates,
            )
        )
        if score > self.best:
            self.best, self.best_step = score, self.last
            self.model.save(self.directory / "best_model")
        write_json(self.directory.parent / "validation_history.json", self.history)
        print(
            f"{identifier(self.case)} step={self.last} validation={score:.4f} best={self.best:.4f}",
            flush=True,
        )

    def _on_step(self):
        if (
            self.num_timesteps - self.last >= self.config["evaluation_every"]
            and self.model._n_updates > 0
        ):
            self.check()
        return True


def train_case(config, case, seed, output_root="results/reference/simulations", quick=False):
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
            Path(__file__).with_name("simulation_envs.py"),
            Path(__file__).with_name("artifacts.py"),
        ],
    )
    run.record["versions"].update({p: version(p) for p in ["minigrid", "mujoco"]})
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir()
    torch.set_num_threads(1)
    env = make_vec_env(
        lambda: make_environment(case["env"], shaping=case["profile"] == "potential"),
        n_envs=case["n_envs"],
        seed=seed,
        monitor_dir=str(checkpoints / "monitor"),
    )
    parameters = dict(
        learning_rate=3e-4,
        seed=seed,
        device=config["device"],
        policy_kwargs=dict(net_arch=[256, 256]),
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
