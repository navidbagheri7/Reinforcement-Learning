"""Replay a public neural actor; optionally save the first seeded episode as a GIF."""

import argparse
from pathlib import Path

import torch

from decision_lab.artifacts import sha256, write_json
from decision_lab.simulation_deployment import load_actor
from decision_lab.simulation_visuals import caption_frames
from decision_lab.simulations import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True, help="Path to the exported actor's NPZ weights")
    parser.add_argument("--seed", type=int, default=300000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--gif", help="Optional output GIF; saves the first seeded episode")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    torch.set_num_threads(1)
    actor = load_actor(args.actor)
    case = actor.metadata["case"]
    result = evaluate(
        actor, case, range(args.seed, args.seed + args.episodes), render=bool(args.gif)
    )
    if args.gif:
        result, frames = result
        path = Path(args.gif)
        path.parent.mkdir(parents=True, exist_ok=True)
        images = caption_frames(
            frames,
            f"{case['env']} | {case['algorithm']} ({case['profile']})",
            f"reset seed {args.seed}",
            1 if case["env"].startswith("MiniGrid") else 4,
            int(result["lengths"][0]),
        )
        duration = 140 if case["env"].startswith("MiniGrid") else 200
        images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0)
        write_json(
            path.with_suffix(".json"),
            dict(
                env=case["env"],
                actor=args.actor,
                actor_sha256=sha256(args.actor),
                test_seed=args.seed,
                return_value=float(result["returns"][0]),
                length=int(result["lengths"][0]),
                gif_sha256=sha256(path),
                protocol="Requested reset seed; no episode selection",
            ),
        )
    print("Episode seeds:", result["seeds"].tolist())
    print("Official returns:", result["returns"].tolist())
    print("Episode lengths:", result["lengths"].tolist())


if __name__ == "__main__":
    main()
