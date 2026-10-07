"""Render the fixed first held-out control episode and verify its saved return."""

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from decision_lab.artifacts import sha256, write_json
from decision_lab.control import ALGORITHMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--algorithm", default="PPO", choices=list(ALGORITHMS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    torch.set_num_threads(1)
    directory = Path("results/reference/control") / args.env / f"{args.algorithm}_seed{args.seed}"
    metrics = json.loads((directory / "metrics.json").read_text())
    checkpoint = directory / "checkpoints/best_model.zip"
    assert sha256(checkpoint) == metrics["checkpoint_sha256"]
    model = ALGORITHMS[args.algorithm].load(checkpoint, device="cpu")
    evidence = np.load(
        Path("assets/source_data/control") / f"{args.env}_{args.algorithm}_{args.seed}.npz"
    )
    test_seed = int(evidence["test_seeds"][0])
    env = gym.make(args.env, render_mode="rgb_array")
    obs, _ = env.reset(seed=test_seed)
    frames = []
    total = 0.0
    length = 0
    while True:
        if length % 4 == 0:
            frames.append(Image.fromarray(env.render()))
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        total += reward
        length += 1
        if term or trunc:
            break
    frames.append(Image.fromarray(env.render()))
    env.close()
    assert total == float(evidence["test_returns"][0]), "Rendering changed the scientific episode"
    output = Path("assets/rollouts") / f"{args.env}_{args.algorithm}_{args.seed}.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True
    )
    write_json(
        output.with_suffix(".json"),
        {
            "env": args.env,
            "algorithm": args.algorithm,
            "fit_seed": args.seed,
            "test_seed": test_seed,
            "return": total,
            "length": length,
            "terminated": term,
            "truncated": trunc,
            "checkpoint_sha256": sha256(checkpoint),
            "gif_sha256": sha256(output),
            "frame_stride": 4,
            "frame_duration_ms": 80,
            "protocol": "Fixed first held-out episode, not visually selected. Return exactly matches saved numeric test evidence.",
        },
    )
    print(output, total, length)


if __name__ == "__main__":
    main()
