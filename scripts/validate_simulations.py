"""Validate original simulation actors, shared dashboards and all rollout evidence."""

import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from decision_lab.artifacts import interval, sha256, write_json
from decision_lab.simulation_deployment import load_actor
from decision_lab.simulations import evaluate, identifier


def main():
    torch.set_num_threads(1)
    checked = []
    for family in ["simulations", "geometry", "curriculum"]:
        config = json.loads(Path("configs", family + ".json").read_text())
        for case in config["cases"]:
            for seed in config["seeds"]:
                directory = Path("results/reference", family) / identifier(case) / str(seed)
                record = json.loads((directory / "run.json").read_text())
                metrics = json.loads((directory / "metrics.json").read_text())
                assert record["status"] == "complete" and not record["config"]["quick"]
                assert record["config"] == dict(benchmark=config, case=case, seed=seed, quick=False)
                for source, digest in record["source_sha256"].items():
                    assert sha256(Path("src/decision_lab") / source) == digest
                assert metrics["actual_steps"] == case["steps"]
                history = json.loads((directory / "validation_history.json").read_text())
                best = max(history, key=lambda entry: entry["mean_return"])
                assert metrics["selected_step"] == best["step"]
                assert metrics["best_validation_return"] == best["mean_return"]
                assert all(row["updates"] > 0 for row in history)
                path = Path("assets/source_data/simulations") / f"{identifier(case)}_{seed}.npz"
                assert sha256(path) == metrics["numeric_sha256"]
                evidence = np.load(path, allow_pickle=False)
                assert np.array_equal(evidence["seeds"], np.arange(100) + 300000)
                assert metrics["test"] == interval(evidence["returns"], seed)
                assert metrics["mean_length"] == evidence["lengths"].mean()
                assert np.all(evidence["terminated"] | evidence["truncated"])
                if metrics["success_fraction"] is not None:
                    assert metrics["success_fraction"] == (evidence["returns"] > 0).mean()
                actor_path = Path("assets/actors") / f"{identifier(case)}_{seed}.npz"
                actor = load_actor(actor_path)
                assert actor.metadata["case"] == case
                assert actor.metadata["checkpoint_sha256"] == metrics["checkpoint_sha256"]
                fresh = evaluate(actor, case, np.arange(3) + 300000)
                for key in ["returns", "lengths", "terminated", "truncated"]:
                    np.testing.assert_array_equal(fresh[key], evidence[key][:3])
                checked.append(str(directory))
    baseline_record = json.loads(Path("docs/simulation_baselines.json").read_text())
    for path, digest in baseline_record["source_sha256"].items():
        assert sha256(path) == digest
    for row in baseline_record["rows"]:
        path = Path("assets/source_data/simulations") / f"{row['env']}_{row['method']}.npz"
        evidence = np.load(path, allow_pickle=False)
        assert sha256(path) == row["numeric_sha256"]
        assert np.array_equal(evidence["seeds"], np.arange(100) + 300000)
        assert interval(evidence["returns"]) == row["test"]
        if row["method"] == "BFS":
            assert row["success_fraction"] == 1.0
    gifs = []
    for path in sorted(Path("assets/rollouts").glob("*.json")):
        metadata = json.loads(path.read_text())
        gif = path.with_suffix(".gif")
        assert sha256(gif) == metadata["gif_sha256"]
        with Image.open(gif) as image:
            assert image.n_frames > 1
        if "source_evidence" in metadata:
            evidence = np.load(metadata["source_evidence"], allow_pickle=False)
            assert metadata["test_seed"] == int(evidence["seeds"][0]) == 300000
            assert metadata["return_value"] == float(evidence["returns"][0])
            assert metadata["length"] == int(evidence["lengths"][0])
        else:
            assert metadata["env"] == "LunarLander-v3"
            evidence = np.load(
                f"assets/source_data/control/{metadata['env']}_{metadata['algorithm']}_{metadata['fit_seed']}.npz",
                allow_pickle=False,
            )
            assert metadata["test_seed"] == int(evidence["test_seeds"][0]) == 300000
            assert metadata["return"] == float(evidence["test_returns"][0])
            assert metadata["length"] == int(evidence["test_lengths"][0])
        gifs.append(str(gif))
    assert len(checked) == 24 and len(baseline_record["rows"]) == 5 and len(gifs) == 48
    dashboard = json.loads(Path("docs/simulation_dashboard.json").read_text())
    assert dashboard["status"] == "passed" and dashboard["fits"] == 42
    assert sha256("docs/simulation_dashboard.html") == dashboard["html_sha256"]
    assert sha256("scripts/templates/simulation_dashboard.html") == dashboard["template_sha256"]
    assert sha256("scripts/create_simulation_dashboard.py") == dashboard["generator_sha256"]
    browser = json.loads(Path("docs/simulation_browser_validation.json").read_text())
    assert browser["status"] == "passed" and len(browser["checked_presets"]) == 14
    assert not browser["javascript_errors"]
    assert browser["html_sha256"] == dashboard["html_sha256"]
    html = Path("docs/simulation_dashboard.html").read_text()
    encoded = re.search(r'<script id="evidence" type="application/json">(.*?)</script>', html, re.S)
    payload = json.loads(encoded.group(1))
    assert len(payload["fits"]) == 42
    for row in payload["fits"]:
        source = json.loads(Path(row["metrics"].removeprefix("../")).read_text())
        assert row["test"] == source["test"] and row["history"] == source["validation_history"]
        assert row["success"] == source["success_fraction"] and row["seed"] == source["seed"]
        assert row["episode"]["gif_sha256"] == sha256(row["gif"].removeprefix("../"))
    assert payload["baselines"] == baseline_record["rows"]
    result = dict(
        status="passed",
        protocols=len(checked),
        actors=len(checked),
        fresh_actor_test_episodes=3 * len(checked),
        unfitted_baselines=5,
        verified_gifs=len(gifs),
        private_checkpoints_required=False,
    )
    write_json("docs/simulation_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
