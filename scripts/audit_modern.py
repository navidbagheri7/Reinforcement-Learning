"""Verify every complete selected model, exported actor and fixed test animation."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from decision_lab.artifacts import sha256, write_json
from decision_lab.modern import ALGORITHMS, evaluate, identifier
from decision_lab.modern_deployment import export
from decision_lab.simulation_visuals import caption_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = []
    files = sorted(Path("results/reference/modern").glob("*/*/metrics.json"))
    assert len(files) == 18 or args.completed_only
    for path in files:
        metrics = json.loads(path.read_text())
        record = json.loads(path.with_name("run.json").read_text())
        assert record["status"] == "complete"
        for name, digest in record["source_sha256"].items():
            assert sha256(Path("src/decision_lab") / name) == digest
        case, seed = record["config"]["case"], metrics["seed"]
        checkpoint = path.parent / "checkpoints/best_model.zip"
        assert sha256(checkpoint) == metrics["checkpoint_sha256"]
        model = ALGORITHMS[case["algorithm"]].load(checkpoint, device="cpu")
        stem = f"{identifier(case)}_{seed}"
        actor_path = Path("assets/modern_actors") / (stem + ".npz")
        actor = export(model, case, actor_path, metrics["checkpoint_sha256"])
        source = Path("assets/source_data/modern") / (stem + ".npz")
        assert sha256(source) == metrics["numeric_sha256"]
        expected = dict(np.load(source, allow_pickle=False))
        fresh = evaluate(actor, case, np.arange(100) + 300000)
        for key in [
            "returns",
            "lengths",
            "terminated",
            "truncated",
            "seeds",
            "first_observations",
            "first_actions",
        ]:
            np.testing.assert_array_equal(fresh[key], expected[key], err_msg=f"{stem}, {key}")
        output = Path(".local/modern_commands") / (stem + ".json")
        subprocess.run(
            [
                sys.executable,
                "scripts/play_modern.py",
                "--actor",
                str(actor_path),
                "--output",
                str(output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        replay = json.loads(output.read_text())
        for key in ["returns", "lengths", "terminated", "truncated"]:
            np.testing.assert_array_equal(replay[key], expected[key][:3])
        rendered, frames = evaluate(actor, case, [300000], render=True)
        np.testing.assert_array_equal(rendered["returns"], expected["returns"][:1])
        stride = 1 if case["profile"] == "partial" else 4
        images = caption_frames(
            frames,
            f"{case['env']} | {case['algorithm']}",
            f"{case['profile']} | fit seed {seed}, test reset 300000",
            stride,
            int(rendered["lengths"][0]),
        )
        gif = Path("assets/rollouts") / (stem + ".gif")
        images[0].save(
            gif,
            save_all=True,
            append_images=images[1:],
            duration=140 if stride == 1 else 200,
            loop=0,
        )
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
                source_evidence=str(source),
                frame_stride=stride,
                protocol="First predetermined held-out episode, no visual selection. Renderer displays the global scene; partial-view actors see only their official local view and direction.",
            ),
        )
        rows.append(
            dict(
                metrics=str(path),
                actor=str(actor_path),
                actor_sha256=sha256(actor_path),
                checkpoint_sha256=metrics["checkpoint_sha256"],
                test_episodes=100,
                fresh_command_episodes=3,
                gif=str(gif),
                gif_sha256=sha256(gif),
            )
        )
        print("Audited", stem, flush=True)
    write_json(
        "docs/modern_actor_audit.json",
        dict(
            status="passed" if len(files) == 18 else "partial",
            actors=rows,
            source_sha256={
                str(p): sha256(p)
                for p in [
                    Path(__file__),
                    Path("scripts/play_modern.py"),
                    Path("src/decision_lab/modern_deployment.py"),
                ]
            },
        ),
    )


if __name__ == "__main__":
    main()
