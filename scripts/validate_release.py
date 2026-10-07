"""Validate portable scientific evidence without private checkpoints or raw coursework."""

import json
import re
from pathlib import Path

import nbformat
import numpy as np

from decision_lab.artifacts import interval, sha256, write_json
from decision_lab.deployment import load_policy
from decision_lab.reporting import records


def strict_read(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant {value} in {path}")

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def main():
    protocols, counts = records()
    for config in Path("configs").glob("*.json"):
        assert (
            config.read_bytes()
            == (Path("src/decision_lab/default_configs") / config.name).read_bytes()
        ), "Bundled defaults differ from experiment configs"
    problems = []
    for family, path, record, metrics in protocols:
        for name, digest in record["source_sha256"].items():
            source = Path("src/decision_lab") / name
            if sha256(source) != digest:
                problems.append(f"Source binding changed: {path}, {name}")
        if record["config"].get("quick", False):
            problems.append(f"Smoke in reference: {path}")
        entries = metrics.get("rows", [metrics])
        for row in entries:
            if "test" not in row:
                continue
            if not re.fullmatch("[0-9a-f]{64}", row["checkpoint_sha256"]):
                problems.append(f"Missing checkpoint digest: {path}")
            if family == "tabular":
                file = (
                    Path("assets/source_data/tabular")
                    / f"{row['env']}_{row['method']}_{row['seed']}.npz"
                )
                key = "returns"
            elif family == "neural":
                file = Path("assets/source_data/neural") / f"{row['recipe']}_{row['seed']}.npz"
                key = "returns"
            elif family == "offline":
                file = (
                    Path("assets/source_data/offline")
                    / f"epsilon{row['behavior_epsilon']:g}_{row['recipe']}_{row['seed']}.npz"
                )
                key = "returns"
            elif family == "control":
                file = (
                    Path("assets/source_data/control")
                    / f"{row['env']}_{row['algorithm']}_{row['seed']}.npz"
                )
                key = "test_returns"
            else:
                file = (
                    Path("assets/source_data/approximation")
                    / f"lambda{row['lambda']:g}_{row['seed']}.npz"
                )
                key = "returns"
            data = np.load(file, allow_pickle=False)
            values = data[key]
            expected = interval(values, row["seed"])
            if len(values) != 100 or not np.isclose(
                expected["mean"], row["test"]["mean"], atol=1e-10, rtol=0
            ):
                problems.append(f"Test evidence disagrees: {path}")
            if not np.allclose(
                expected["episode_bootstrap_ci95"],
                row["test"]["episode_bootstrap_ci95"],
                atol=1e-10,
                rtol=0,
            ):
                problems.append(f"Bootstrap evidence disagrees: {path}")
            seeds = data["test_seeds" if family == "control" else "seeds"]
            if not np.array_equal(seeds, np.arange(100) + 300000):
                problems.append(f"Wrong test seeds: {file}")
            if family in ["tabular", "neural"]:
                budget = (
                    record["config"]["budgets"][row["env"]]
                    if family == "tabular"
                    else record["config"]["steps"]
                )
                if data["state_action_visits"].sum() != budget:
                    problems.append(f"Interaction budget disagrees: {file}")
            if family == "offline":
                logged = (
                    Path("assets/source_data/offline")
                    / f"logged_epsilon{row['behavior_epsilon']:g}_{row['seed']}.npz"
                )
                if sha256(logged) != row["dataset_sha256"]:
                    problems.append(f"Logged dataset changed: {logged}")
    numeric = list(Path("assets").rglob("*.npz"))
    for file in numeric:
        with np.load(file, allow_pickle=False) as data:
            for name in data.files:
                if not np.isfinite(data[name]).all():
                    problems.append(f"Nonfinite array: {file}, {name}")
    policies = list(Path("assets/policies").glob("*.json"))
    if len(policies) != 108:
        problems.append(f"Expected 108 finite policies, got {len(policies)}")
    for file in policies:
        load_policy(file)
    notebooks = list(Path("notebooks").glob("*.ipynb"))
    code_count = 0
    if len(notebooks) != 11:
        problems.append(f"Expected 11 walkthroughs, got {len(notebooks)}")
    for file in notebooks:
        notebook = nbformat.read(file, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            code_count += 1
            if cell.execution_count is None or any(
                out.output_type == "error" for out in cell.outputs
            ):
                problems.append(f"Unexecuted/failed notebook: {file}")
    for file in Path("assets/figures").glob("*.png"):
        if not file.with_suffix(".pdf").exists():
            problems.append(f"Missing PDF: {file}")
    result = {
        "status": "passed" if not problems else "failed",
        "reference_protocols": counts,
        "numeric_artifacts": len(numeric),
        "finite_policies": len(policies),
        "notebooks": len(notebooks),
        "executed_code_cells": code_count,
        "problems": problems,
    }
    write_json("docs/validation.json", result)
    if problems:
        raise ValueError("\n".join(problems))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
