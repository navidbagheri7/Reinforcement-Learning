"""Maintained-library control comparisons with isolated validation and deployment.

Fixed repeated validation seeds select checkpoints. Official-task test seeds and
shift seeds are used only after fitting. Time-limit bootstrap is handled by SB3.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from .artifacts import Run, interval, sha256, write_json

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ALGORITHMS = {"DQN": DQN, "PPO": PPO, "SAC": SAC}


def evaluate(model, env_id, seeds, kwargs=None):
    env = gym.make(env_id, **(kwargs or {}))
    returns = []
    lengths = []
    terminated = []
    truncated = []
    first_observations = []
    first_actions = []
    for index, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        total = 0.0
        length = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            if index == 0 and length < 32:
                first_observations.append(np.asarray(obs).copy())
                first_actions.append(np.asarray(action).copy())
            obs, r, term, trunc, _ = env.step(action)
            total += r
            length += 1
            if term or trunc:
                break
        returns.append(total)
        lengths.append(length)
        terminated.append(term)
        truncated.append(trunc)
    env.close()
    return {
        "returns": np.array(returns),
        "lengths": np.array(lengths),
        "terminated": np.array(terminated),
        "truncated": np.array(truncated),
        "seeds": np.array(seeds),
        "first_observations": np.array(first_observations),
        "first_actions": np.array(first_actions),
    }


class ValidationCallback(BaseCallback):
    def __init__(self, env_id, episodes, every, path):
        super().__init__()
        self.env_id, self.episodes, self.every, self.path = env_id, episodes, every, Path(path)
        self.history = []
        self.best = -np.inf
        self.best_step = None
        self.next_evaluation = every

    def check(self):
        evidence = evaluate(self.model, self.env_id, np.arange(self.episodes) + 200000)
        score = float(evidence["returns"].mean())
        row = {
            "step": self.num_timesteps,
            "mean_return": score,
            "returns": evidence["returns"].tolist(),
            "optimizer_updates": int(self.model._n_updates),
        }
        self.history.append(row)
        if score > self.best:
            self.best = score
            self.best_step = self.num_timesteps
            self.model.save(self.path / "best_model")
        write_json(self.path.parent / "validation_history.json", self.history)
        print(
            self.env_id,
            self.model.__class__.__name__,
            "step",
            self.num_timesteps,
            "validation",
            score,
            flush=True,
        )

    def _on_step(self):
        if self.num_timesteps >= self.next_evaluation and self.model._n_updates > 0:
            self.check()
            self.next_evaluation += self.every
        return True


def shift_spec(env_id):
    if env_id == "LunarLander-v3":
        return {"gravity": -11.0, "enable_wind": True, "wind_power": 10.0, "turbulence_power": 1.0}
    if env_id == "Pendulum-v1":
        return {"g": 12.0}
    return None


def train_case(config, case, seed, output_root="results/reference/control", quick=False):
    torch.set_num_threads(1)
    name = f"{case['env']}/{case['algorithm']}_seed{seed}"
    directory = Path(output_root) / name
    if (directory / "run.json").exists():
        old = json.loads((directory / "run.json").read_text())
        if old["status"] == "complete" and old["config"] == {
            "benchmark": config,
            "case": case,
            "seed": seed,
            "quick": quick,
        }:
            print("Already complete", name, flush=True)
            return
        raise RuntimeError(
            f"Existing incomplete or different run at {directory}; preserve it and use a new output directory"
        )
    run = Run(
        directory,
        {"benchmark": config, "case": case, "seed": seed, "quick": quick},
        [__file__, Path(__file__).with_name("artifacts.py")],
    )
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir()
    env = make_vec_env(
        case["env"], n_envs=case["n_envs"], seed=seed, monitor_dir=str(checkpoints / "training")
    )
    parameters = {
        "learning_rate": case["learning_rate"],
        "policy_kwargs": {"net_arch": case["net"]},
        "seed": seed,
        "device": config["device"],
        "verbose": 0,
    }
    if case["algorithm"] == "PPO":
        parameters.update(
            n_steps=256,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01 if case["env"] == "LunarLander-v3" else 0.0,
            clip_range=0.2,
        )
    else:
        parameters.update(
            buffer_size=100000,
            batch_size=128,
            learning_starts=1000 if quick else 10000,
            train_freq=4,
            gradient_steps=1,
            gamma=0.99,
        )
        if case["algorithm"] == "DQN":
            parameters.update(
                exploration_fraction=0.25, exploration_final_eps=0.05, target_update_interval=1000
            )
        else:
            parameters.update(tau=0.005, ent_coef="auto")
    model = ALGORITHMS[case["algorithm"]]("MlpPolicy", env, **parameters)
    callback = ValidationCallback(
        case["env"], config["validation_episodes"], config["evaluation_every"], checkpoints
    )
    model.learn(total_timesteps=case["steps"], callback=callback)
    # Callback steps occur during collection, before the final optimizer update.
    # Evaluate the actual fitted model once learning returns, even at the same step.
    callback.num_timesteps = model.num_timesteps
    callback.model = model
    callback.check()
    model.save(checkpoints / "final_model")
    selected = ALGORITHMS[case["algorithm"]].load(
        checkpoints / "best_model.zip", device=config["device"]
    )
    test = evaluate(selected, case["env"], np.arange(config["test_episodes"]) + 300000)
    shift = None
    kwargs = shift_spec(case["env"])
    if kwargs:
        shift = evaluate(
            selected, case["env"], np.arange(config["shift_episodes"]) + 400000, kwargs
        )
    public = directory / "source_data" if quick else Path("assets/source_data/control")
    public.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        public / f"{case['env']}_{case['algorithm']}_{seed}.npz",
        **{"test_" + k: v for k, v in test.items()},
        **({} if shift is None else {"shift_" + k: v for k, v in shift.items()}),
    )
    training = []
    for file in sorted((checkpoints / "training").glob("*.monitor.csv")):
        with file.open() as stream:
            stream.readline()
            rows = list(csv.DictReader(stream))
        endings = np.cumsum([int(row["l"]) for row in rows]) * case["n_envs"]
        for row, step in zip(rows, endings, strict=True):
            training.append(
                (int(step), float(row["r"]), int(row["l"]), int(file.name.split(".")[0]))
            )
    training.sort(key=lambda row: row[0])
    np.savez_compressed(
        public / f"{case['env']}_{case['algorithm']}_{seed}_training.npz",
        episodes=np.asarray(training, dtype=float).reshape(-1, 4),
    )
    metadata = {
        "algorithm": case["algorithm"],
        "env": case["env"],
        "seed": seed,
        "selected_step": callback.best_step,
        "checkpoint_sha256": sha256(checkpoints / "best_model.zip"),
        "gymnasium_version": gym.__version__,
        "protocol": "Deterministic selected-policy deployment. Validation seeds 200000.., final test300000.., shift400000... Training seeds start at fit seed with vector-environment offsets. Local reference uses CPU float32, no reward/observation normalization.",
    }
    write_json(directory / "inference.json", metadata)
    result = {
        "env": case["env"],
        "algorithm": case["algorithm"],
        "seed": seed,
        "actual_steps": model.num_timesteps,
        "selected_step": callback.best_step,
        "best_validation_return": callback.best,
        "test": interval(test["returns"], seed),
        "test_termination_fraction": float(test["terminated"].mean()),
        "test_truncation_fraction": float(test["truncated"].mean()),
        "test_success_fraction": float((test["returns"] >= 200).mean())
        if case["env"] == "LunarLander-v3"
        else None,
        "shift_kwargs": kwargs,
        "shift": None if shift is None else interval(shift["returns"], seed),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "validation_history": callback.history,
        "library_hyperparameters": parameters,
        "framework": "stable-baselines3 2.8.0; these are library implementations, not owned algorithm internals",
    }
    env.close()
    run.finish(result)
    print("Complete", name, result["test"]["mean"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/control.json")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--case", type=int)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-root", default="results/reference/control")
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    cases = config["cases"] if a.case is None else [config["cases"][a.case]]
    for case in cases:
        if a.quick:
            config = config | {
                "validation_episodes": 2,
                "test_episodes": 3,
                "shift_episodes": 2,
                "evaluation_every": 2048,
            }
            case = case | {"steps": 2048}
        train_case(config, case, a.seed, a.output_root, a.quick)


if __name__ == "__main__":
    main()
