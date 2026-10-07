"""Matched-interaction control and partially observed recurrent navigation.

These use the maintained SB3 implementations. Validation resets select policies;
the independent official test episodes are evaluated only after fitting.
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
from sb3_contrib import TQC, RecurrentPPO
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

from .artifacts import Run, interval, sha256, write_json

ALGORITHMS = {"TD3": TD3, "TQC": TQC, "PPO": PPO, "RecurrentPPO": RecurrentPPO}


class PartialObservation(gym.ObservationWrapper):
    """Official egocentric 7x7 symbolic view and direction, without global pose.

    Object, colour and state codes are scaled by their declared maxima. The
    constant task instruction is omitted. Hidden map, goal coordinates, carrying
    state and elapsed time are never queried to construct this observation.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1, shape=(151,), dtype=np.float32)

    def observation(self, obs):
        image = np.asarray(obs["image"], np.float32)
        if image.shape != (7, 7, 3):
            raise ValueError("The partial-view protocol requires the official 7x7 view")
        return np.concatenate(
            [(image / [10, 5, 2]).reshape(-1), np.eye(4)[int(obs["direction"])]]
        ).astype(np.float32)


def environment(case, render=False):
    env = gym.make(case["env"], **({"render_mode": "rgb_array"} if render else {}))
    return PartialObservation(env) if case["profile"] == "partial" else env


def identifier(case):
    return f"{case['env']}_{case['algorithm']}_{case['profile']}"


def evaluate(model, case, seeds, render=False):
    """Reset memory on every new episode; retain memory across its transitions."""
    env = environment(case, render)
    arrays = {key: [] for key in ["returns", "lengths", "terminated", "truncated"]}
    observations, actions, frames = [], [], []
    try:
        for index, seed in enumerate(seeds):
            obs, _ = env.reset(seed=int(seed))
            memory, start, total, length = None, np.array([True]), 0.0, 0
            while True:
                action, memory = model.predict(
                    obs, state=memory, episode_start=start, deterministic=True
                )
                start[:] = False
                if index == 0 and length < 32:
                    observations.append(np.asarray(obs).copy())
                    actions.append(np.asarray(action).copy())
                if (
                    render
                    and index == 0
                    and length % (1 if case["profile"] == "partial" else 4) == 0
                ):
                    frames.append(env.render())
                obs, reward, term, trunc, _ = env.step(action)
                total += reward
                length += 1
                if term or trunc:
                    for key, value in zip(arrays, [total, length, term, trunc], strict=True):
                        arrays[key].append(value)
                    if render and index == 0:
                        frames.append(env.render())
                    break
    finally:
        env.close()
    evidence = {key: np.asarray(value) for key, value in arrays.items()}
    evidence.update(
        seeds=np.asarray(seeds),
        first_observations=np.asarray(observations),
        first_actions=np.asarray(actions),
    )
    return (evidence, frames) if render else evidence


class Selection(BaseCallback):
    def __init__(self, case, config, directory):
        super().__init__()
        self.case, self.config, self.directory = case, config, directory
        self.history, self.best, self.selected = [], -np.inf, None
        self.next_evaluation = config["evaluation_every"]

    def check(self):
        values = evaluate(
            self.model, self.case, np.arange(self.config["validation_episodes"]) + 200000
        )
        score = float(values["returns"].mean())
        self.history.append(
            dict(
                step=self.model.num_timesteps,
                updates=int(self.model._n_updates),
                returns=values["returns"].tolist(),
                mean_return=score,
            )
        )
        if score > self.best:
            self.best, self.selected = score, self.model.num_timesteps
            self.model.save(self.directory / "best_model")
        write_json(self.directory.parent / "validation_history.json", self.history)
        print(identifier(self.case), self.model.num_timesteps, "validation", score, flush=True)

    def _on_step(self):
        if self.num_timesteps >= self.next_evaluation and self.model._n_updates > 0:
            self.check()
            self.next_evaluation += self.config["evaluation_every"]
        return True


def train(config, case, seed, output="results/reference/modern", quick=False):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources = [Path(__file__), Path(__file__).with_name("artifacts.py")]
    bound = {p.name: sha256(p) for p in sources}
    directory = Path(output) / identifier(case) / str(seed)
    protocol = dict(benchmark=config, case=case, seed=seed, quick=quick)
    if (directory / "run.json").exists():
        old = json.loads((directory / "run.json").read_text())
        if (
            old["status"] == "complete"
            and old["config"] == protocol
            and old["source_sha256"] == bound
        ):
            print("Already complete", directory, flush=True)
            return
        raise ValueError("Preserve the incomplete/different run; choose a fresh output directory")
    run = Run(directory, protocol, sources)
    run.record["versions"].update({p: version(p) for p in ["sb3-contrib", "minigrid", "mujoco"]})
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir()
    env = make_vec_env(
        lambda: environment(case),
        n_envs=case["n_envs"],
        seed=seed,
        monitor_dir=str(checkpoints / "monitor"),
    )
    parameters = dict(learning_rate=3e-4, gamma=0.99, seed=seed, device=config["device"], verbose=0)
    if case["profile"] == "partial":
        parameters.update(
            n_steps=256,
            batch_size=256,
            n_epochs=10,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[128, 128]),
        )
        if case["algorithm"] == "RecurrentPPO":
            parameters["policy_kwargs"].update(
                lstm_hidden_size=128, n_lstm_layers=1, shared_lstm=False, enable_critic_lstm=True
            )
    else:
        parameters.update(
            buffer_size=1000000,
            batch_size=256,
            learning_starts=10000 if not quick else 128,
            train_freq=4,
            gradient_steps=1,
            tau=0.005,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
        if case["algorithm"] == "TD3":
            parameters.update(
                action_noise=NormalActionNoise(np.zeros(6), 0.1 * np.ones(6)),
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
            )
        else:
            parameters.update(ent_coef="auto", top_quantiles_to_drop_per_net=2)
            parameters["policy_kwargs"].update(n_critics=2, n_quantiles=25)
    algorithm = ALGORITHMS[case["algorithm"]]
    model = algorithm(
        "MlpLstmPolicy" if case["algorithm"] == "RecurrentPPO" else "MlpPolicy", env, **parameters
    )
    callback = Selection(case, config, checkpoints)
    try:
        model.learn(case["steps"], callback=callback)
        callback.check()  # the final optimizer update occurs after collection callbacks
        model.save(checkpoints / "final_model")
        selected = algorithm.load(checkpoints / "best_model.zip", device="cpu")
        evidence = evaluate(selected, case, np.arange(config["test_episodes"]) + 300000)
    finally:
        env.close()
    public = directory / "source_data" if quick else Path("assets/source_data/modern")
    public.mkdir(parents=True, exist_ok=True)
    numeric = public / f"{identifier(case)}_{seed}.npz"
    np.savez_compressed(numeric, **evidence)
    # Noise is a training-only SB3 object; save its declared normalized-action SD.
    declared = dict(parameters)
    if "action_noise" in declared:
        declared["action_noise"] = dict(kind="NormalActionNoise", normalized_sigma=0.1)
    run.finish(
        dict(
            env=case["env"],
            algorithm=case["algorithm"],
            profile=case["profile"],
            seed=seed,
            actual_steps=model.num_timesteps,
            optimizer_updates=int(model._n_updates),
            selected_step=callback.selected,
            validation_history=callback.history,
            best_validation_return=callback.best,
            test=interval(evidence["returns"], seed),
            mean_length=float(evidence["lengths"].mean()),
            success_fraction=float(evidence["terminated"].mean())
            if case["profile"] == "partial"
            else None,
            checkpoint_sha256=sha256(checkpoints / "best_model.zip"),
            numeric_sha256=sha256(numeric),
            parameters=declared,
            protocol="Official sparse reward and dynamics. Validation 200000..200011 selects first best mean; test 300000..300099 once after fitting. Recurrent state resets at each episode. Partial-view PPO and recurrent PPO share observations and interaction budgets, but have different parameter/update costs. TD3/TQC match the earlier HalfCheetah SAC interaction/replay/update presets; device and critic cost differ. Maintained SB3 2.8.0 implementations; no paper benchmark reproduction claim.",
        )
    )
    print("COMPLETE", identifier(case), seed, float(evidence["returns"].mean()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/modern.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--case", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output", default="results/reference/modern")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.device:
        config["device"] = args.device
    if args.quick:
        config.update(validation_episodes=2, test_episodes=3, evaluation_every=2048)
    for case in config["cases"] if args.case is None else [config["cases"][args.case]]:
        train(
            config,
            case | {"steps": 2048} if args.quick else case,
            args.seed,
            args.output,
            args.quick,
        )


if __name__ == "__main__":
    main()
