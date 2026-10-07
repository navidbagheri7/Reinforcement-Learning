"""Summarize all new fits, success rates and descriptive validation milestones."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from decision_lab.artifacts import write_json


def main():
    rows = []
    for family in ["simulations", "geometry", "curriculum"]:
        for path in sorted(Path("results/reference", family).glob("*/*/metrics.json")):
            m = json.loads(path.read_text())
            history = m["validation_history"]
            first = next((r["step"] for r in history if r["mean_return"] >= 0.9), None)
            planner_gap = None
            if m["env"].startswith("MiniGrid"):
                root = Path("assets/source_data/simulations")
                policy_data = np.load(
                    root / f"{m['env']}_{m['algorithm']}_{m['profile']}_{m['seed']}.npz"
                )
                planner_data = np.load(root / f"{m['env']}_BFS.npz")
                np.testing.assert_array_equal(policy_data["seeds"], planner_data["seeds"])
                differences = planner_data["returns"] - policy_data["returns"]
                assert differences.min() >= -1e-12
                planner_gap = float(differences.mean())
            rows.append(
                dict(
                    env=m["env"],
                    algorithm=m["algorithm"],
                    profile=m["profile"],
                    seed=m["seed"],
                    mean_test_return=m["test"]["mean"],
                    success=m["success_fraction"],
                    mean_length=m["mean_length"],
                    mean_planner_episode_return_gap=planner_gap,
                    selected_step=m["selected_step"],
                    first_validation_return_at_least_0_9=first
                    if m["env"].startswith("MiniGrid")
                    else None,
                    source_metrics=str(path),
                )
            )
    assert len(rows) == 24
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["env", "algorithm", "profile"]).agg(
        fits=("seed", "size"),
        mean_return=("mean_test_return", "mean"),
        fit_sd=("mean_test_return", "std"),
        worst_fit_return=("mean_test_return", "min"),
        goal_success=("success", "mean"),
        worst_fit_goal_success=("success", "min"),
        mean_length=("mean_length", "mean"),
        mean_planner_episode_return_gap=("mean_planner_episode_return_gap", "mean"),
    )
    grouped.to_csv("docs/simulation_comparison.csv")
    write_json(
        "docs/simulation_analysis.json",
        dict(
            rows=rows,
            scope="Descriptive analysis after fitting; no checkpoint/hyperparameter selection",
            milestone="First observed mean official validation return >=0.9; not sustained success or a test guarantee",
            training_variance="SD of 3 independently fitted test means; conditional episode CIs are in each run",
            task_returns_comparable_only_within_task=True,
        ),
    )
    print(grouped.to_string())
    assert np.isfinite(frame.mean_test_return).all()


if __name__ == "__main__":
    main()
