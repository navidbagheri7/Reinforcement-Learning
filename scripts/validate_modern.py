"""Recompute extension statistics and replay every exported actor without checkpoints."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from decision_lab.artifacts import interval, sha256, write_json
from decision_lab.modern import identifier
from decision_lab.modern_deployment import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 100:
        parser.error("Replay between 1 and 100 fixed held-out episodes")
    config = json.loads(Path("configs/modern.json").read_text())
    files = sorted(Path("results/reference/modern").glob("*/*/metrics.json"))
    assert len(files) == 18
    checked = []
    with tempfile.TemporaryDirectory(prefix="decision-lab-modern-") as directory:
        for path in files:
            metrics = json.loads(path.read_text())
            run = json.loads(path.with_name("run.json").read_text())
            case = run["config"]["case"]
            seed = metrics["seed"]
            assert run["status"] == "complete" and not run["config"]["quick"]
            assert case in config["cases"] and seed in [0, 1, 2]
            assert metrics["actual_steps"] == case["steps"] == 1048576
            assert metrics["optimizer_updates"] > 0
            history = metrics["validation_history"]
            best = max(history, key=lambda row: row["mean_return"])
            assert metrics["selected_step"] == best["step"]
            assert best["mean_return"] == metrics["best_validation_return"]
            assert all(row["updates"] > 0 and len(row["returns"]) == 12 for row in history)
            assert all(
                np.isclose(np.mean(row["returns"]), row["mean_return"], atol=1e-12)
                for row in history
            )
            for name, value in run["source_sha256"].items():
                assert sha256(Path("src/decision_lab") / name) == value
            stem = f"{identifier(case)}_{seed}"
            source = Path("assets/source_data/modern") / (stem + ".npz")
            actor_path = Path("assets/modern_actors") / (stem + ".npz")
            assert sha256(source) == metrics["numeric_sha256"]
            actor = load(actor_path)
            assert actor.metadata["case"] == case
            assert actor.metadata["checkpoint_sha256"] == metrics["checkpoint_sha256"]
            with np.load(source, allow_pickle=False) as data:
                evidence = dict(data)
            assert evidence["returns"].shape == (100,)
            np.testing.assert_array_equal(evidence["seeds"], np.arange(100) + 300000)
            expected = interval(evidence["returns"], seed)
            for key in ["mean", "episode_bootstrap_ci95"]:
                np.testing.assert_allclose(expected[key], metrics["test"][key], rtol=0, atol=1e-10)
            if case["profile"] == "partial":
                assert metrics["success_fraction"] == float(evidence["terminated"].mean())
            gif = Path("assets/rollouts") / (stem + ".gif")
            caption = json.loads(gif.with_suffix(".json").read_text())
            assert caption["gif_sha256"] == sha256(gif)
            assert caption["actor_sha256"] == sha256(actor_path)
            assert caption["test_seed"] == 300000 and caption["fit_seed"] == seed
            assert caption["return_value"] == evidence["returns"][0]
            assert caption["length"] == evidence["lengths"][0]
            output = Path(directory) / (stem + ".json")
            subprocess.run(
                [
                    sys.executable,
                    "scripts/play_modern.py",
                    "--actor",
                    str(actor_path),
                    "--episodes",
                    str(args.episodes),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                timeout=180,
            )
            fresh = json.loads(output.read_text())
            for key in ["returns", "lengths", "terminated", "truncated", "seeds"]:
                np.testing.assert_array_equal(fresh[key], evidence[key][: args.episodes])
            checked.append(dict(model=stem, replayed_episodes=args.episodes))
            print("Verified portable", stem, flush=True)
    write_json(
        "docs/modern_validation.json",
        dict(
            status="passed",
            protocols=18,
            fits=checked,
            scope="Independent selected-step/100-episode statistics, source/numeric/actor/GIF identities and fresh CPU deployment commands. Episodic memory reset/persistence is validated by the portable-actor tests. Private selected models are audited separately over all 100 episodes.",
        ),
    )


if __name__ == "__main__":
    main()
