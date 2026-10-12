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
            observations, actions = evidence["first_observations"], evidence["first_actions"]
            assert len(observations) == len(actions) > 0
            memory = None
            predicted = []
            for index, observation in enumerate(observations):
                action, memory = actor.predict(
                    observation,
                    state=memory,
                    episode_start=np.asarray([index == 0]),
                    deterministic=True,
                )
                predicted.append(np.asarray(action))
            predicted = np.asarray(predicted)
            continuous = case["algorithm"] in {"TD3", "TQC"}
            if continuous:
                np.testing.assert_allclose(predicted, actions, rtol=2e-6, atol=5e-6)
            else:
                np.testing.assert_array_equal(predicted, actions)
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
            np.testing.assert_array_equal(
                np.asarray(fresh["first_observations"])[0], observations[0]
            )
            assert np.isfinite(fresh["returns"]).all()
            for key in ["lengths", "terminated", "truncated", "seeds"]:
                np.testing.assert_array_equal(fresh[key], evidence[key][: args.episodes])
            if not continuous:
                np.testing.assert_array_equal(
                    fresh["returns"], evidence["returns"][: args.episodes]
                )
            checked.append(
                dict(
                    model=stem,
                    replayed_episodes=args.episodes,
                    fixed_observation_action_checks=len(observations),
                    continuous_physics=continuous,
                    archived_returns=evidence["returns"][: args.episodes].tolist(),
                    fresh_returns=fresh["returns"],
                    maximum_return_difference=float(
                        np.max(
                            np.abs(
                                np.asarray(fresh["returns"]) - evidence["returns"][: args.episodes]
                            )
                        )
                    ),
                    maximum_fixed_action_difference=float(np.max(np.abs(predicted - actions))),
                )
            )
            print("Verified portable", stem, flush=True)
    write_json(
        "docs/modern_validation.json",
        dict(
            status="passed",
            protocols=18,
            fits=checked,
            fixed_observation_action_checks=sum(
                row["fixed_observation_action_checks"] for row in checked
            ),
            scope="Independent selected-step/100-episode statistics, source/numeric/actor/GIF identities, fixed-observation action checks with episodic memory, and fresh CPU deployment commands. Continuous actions use float32 precision tolerances; their fresh physics returns are separate diagnostic replays, not bit-identical benchmark reproductions or replacements for archived scores. Discrete outputs/returns, initial states, episode lengths and termination flags remain exact. Episodic memory reset/persistence is validated by portable-actor tests; original selected models were audited separately over all 100 episodes.",
        ),
    )


if __name__ == "__main__":
    main()
