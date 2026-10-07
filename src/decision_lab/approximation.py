"""Owned tile-coded linear SARSA(lambda) on official MountainCar-v0."""

from pathlib import Path

import gymnasium as gym
import numpy as np

from .artifacts import Run, interval, sha256


class Tiles:
    def __init__(self, tilings=8, bins=8):
        self.tilings, self.bins = tilings, bins
        self.size = 3 * tilings * (bins + 1) ** 2
        self.offset = np.array(
            [(t / tilings, ((3 * t) % tilings) / tilings) for t in range(tilings)]
        )

    def indices(self, state, action):
        if not 0 <= action < 3:
            raise ValueError("MountainCar action must be 0..2")
        normalized = (np.asarray(state) - [-1.2, -0.07]) / [1.8, 0.14]
        coordinates = np.clip(
            np.floor(normalized * self.bins + self.offset).astype(int), 0, self.bins
        )
        return (
            action * self.tilings * (self.bins + 1) ** 2
            + np.arange(self.tilings) * (self.bins + 1) ** 2
            + coordinates[:, 0] * (self.bins + 1)
            + coordinates[:, 1]
        )

    def q(self, weights, state):
        return np.array([weights[self.indices(state, a)].sum() for a in range(3)])


def rollout(weights, tiles, seeds):
    env = gym.make("MountainCar-v0")
    returns = []
    lengths = []
    success = []
    trace = []
    for index, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        total = 0.0
        length = 0
        while True:
            action = int(tiles.q(weights, obs).argmax())
            if index == 0 and length < 32:
                trace.append((*obs, action))
            obs, r, term, trunc, _ = env.step(action)
            total += r
            length += 1
            if term or trunc:
                break
        returns.append(total)
        lengths.append(length)
        success.append(term)
    env.close()
    return {
        "returns": np.array(returns),
        "lengths": np.array(lengths),
        "success": np.array(success),
        "first_trace": np.array(trace),
        "seeds": np.array(seeds),
    }


def fit(config, seed, output="results/reference/approximation", quick=False):
    root = Path(output) / f"seed{seed}"
    run = Run(root, config | {"seed": seed, "quick": quick}, [__file__])
    destination = (
        run.directory / "source_data" if quick else Path("assets/source_data/approximation")
    )
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for lam in config["lambdas"]:
        tiles = Tiles()
        weights = np.zeros(tiles.size)
        rng = np.random.default_rng(seed)
        env = gym.make("MountainCar-v0")
        best = -np.inf
        selected = 0
        best_weights = None
        history = []
        training = []
        steps = 0

        def choose(obs):
            return (
                int(rng.integers(3))
                if rng.random() < config["epsilon"]
                else int(tiles.q(weights, obs).argmax())
            )

        for episode in range(1, config["episodes"] + 1):
            obs, _ = env.reset(seed=seed if episode == 1 else None)
            action = choose(obs)
            eligibility = np.zeros_like(weights)
            total = 0
            while True:
                nxt, reward, term, trunc, _ = env.step(action)
                next_action = choose(nxt)
                indices = tiles.indices(obs, action)
                delta = (
                    reward
                    + (0.0 if term else weights[tiles.indices(nxt, next_action)].sum())
                    - weights[indices].sum()
                )
                eligibility *= lam
                eligibility[indices] = 1.0
                weights += config["alpha"] / tiles.tilings * delta * eligibility
                steps += 1
                total += reward
                obs, action = nxt, next_action
                if term or trunc:
                    break
            training.append((episode, steps, total))
            if episode % config["validation_every"] == 0 or episode == config["episodes"]:
                evidence = rollout(
                    weights, tiles, np.arange(config["validation_episodes"]) + 200000
                )
                score = float(evidence["returns"].mean())
                history.append({"episode": episode, "steps": steps, "return": score})
                if score > best:
                    best = score
                    best_weights = weights.copy()
                    selected = episode
                print("SARSA tiles", lam, seed, episode, score, flush=True)
        env.close()
        evidence = rollout(best_weights, tiles, np.arange(config["test_episodes"]) + 300000)
        name = f"lambda{lam:g}_{seed}"
        np.savez_compressed(
            destination / f"{name}.npz", **evidence, training_episodes=np.array(training)
        )
        checkpoint = root / "checkpoints" / f"{name}.npz"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            checkpoint, weights=best_weights, tilings=tiles.tilings, bins=tiles.bins
        )
        results.append(
            {
                "lambda": lam,
                "seed": seed,
                "training_steps": steps,
                "selected_episode": selected,
                "test": interval(evidence["returns"], seed),
                "success_fraction": float(evidence["success"].mean()),
                "validation_history": history,
                "checkpoint_sha256": sha256(checkpoint),
            }
        )
    run.finish(
        {
            "rows": results,
            "protocol": "Official MountainCar-v0, 200-step time limit. Physical bounded 8-tiling 8-bin features, replacing traces, SARSA, gamma1, epsilon .05, effective step alpha/8. True termination zeroes bootstrap; truncation retains it, then traces reset with the episode. Equal episode budgets have different interaction counts, which are reported. Validation-only selection, independent fixed test resets.",
        }
    )
