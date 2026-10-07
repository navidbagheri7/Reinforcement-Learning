"""Export every selected actor, audit held-out inference, and render fixed test GIFs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from decision_lab.artifacts import sha256, write_json
from decision_lab.simulation_deployment import export_actor
from decision_lab.simulation_visuals import caption_frames
from decision_lab.simulations import ALGORITHMS, evaluate, identifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Audit currently complete fits and mark output partial",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    audited = []
    metrics_files = [
        path
        for family in ["simulations", "geometry", "curriculum"]
        for path in Path("results/reference", family).glob("*/*/metrics.json")
    ]
    for path in sorted(metrics_files):
        metrics = json.loads(path.read_text())
        record = json.loads(path.with_name("run.json").read_text())
        assert record["status"] == "complete"
        case, seed = record["config"]["case"], metrics["seed"]
        checkpoint = path.parent / "checkpoints/best_model.zip"
        assert sha256(checkpoint) == metrics["checkpoint_sha256"]
        model = ALGORITHMS[case["algorithm"]].load(checkpoint, device="cpu")
        actor_path = Path("assets/actors") / f"{identifier(case)}_{seed}.npz"
        actor = export_actor(model, case, actor_path, metrics["checkpoint_sha256"])
        evidence = np.load(f"assets/source_data/simulations/{identifier(case)}_{seed}.npz")
        for obs in evidence["first_observations"]:
            original = model.predict(obs, deterministic=True)[0]
            exported = actor.predict(obs)[0]
            np.testing.assert_array_equal(exported, original)
        fresh = evaluate(actor, case, np.arange(3) + 300000)
        for key in ["returns", "lengths", "terminated", "truncated"]:
            np.testing.assert_array_equal(fresh[key], evidence[key][:3])
        audited.append(
            dict(
                actor=str(actor_path),
                actor_sha256=sha256(actor_path),
                checkpoint_sha256=metrics["checkpoint_sha256"],
                fresh_test_episodes=3,
            )
        )
        rendered, frames = evaluate(actor, case, [300000], render=True)
        np.testing.assert_array_equal(rendered["returns"], evidence["returns"][:1])
        images = caption_frames(
            frames,
            f"{case['env']} | {case['algorithm']} ({case['profile']})",
            f"fit seed {seed}, test reset 300000",
            1 if case["env"].startswith("MiniGrid") else 4,
            int(rendered["lengths"][0]),
        )
        gif = Path("assets/rollouts") / f"{identifier(case)}_{seed}.gif"
        duration = 140 if case["env"].startswith("MiniGrid") else 200
        images[0].save(gif, save_all=True, append_images=images[1:], duration=duration, loop=0)
        write_json(
            gif.with_suffix(".json"),
            dict(
                env=case["env"],
                algorithm=case["algorithm"],
                profile=case["profile"],
                fit_seed=seed,
                test_seed=300000,
                return_value=float(rendered["returns"][0]),
                length=int(rendered["lengths"][0]),
                terminated=bool(rendered["terminated"][0]),
                truncated=bool(rendered["truncated"][0]),
                actor_sha256=sha256(actor_path),
                gif_sha256=sha256(gif),
                frame_stride=1 if case["env"].startswith("MiniGrid") else 4,
                frame_duration_ms=duration,
                source_evidence=f"assets/source_data/simulations/{identifier(case)}_{seed}.npz",
                protocol="Fixed first held-out episode; exported actor return exactly matches numeric test evidence; no visual selection",
            ),
        )
        print("Audited", identifier(case), seed, flush=True)
    assert args.allow_partial or len(audited) == 24, (
        "Expected all three fitting seeds for all eight declared protocols"
    )
    write_json(
        "docs/simulation_actor_audit.json",
        dict(status="passed" if len(audited) == 24 else "partial", actors=audited),
    )


if __name__ == "__main__":
    main()
