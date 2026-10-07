"""Build the visual gallery and success table from measured evidence."""

import json
from pathlib import Path

import pandas as pd


def animation(prose, stem, description, seeds):
    prose += [f"### {description}", ""]
    for seed in seeds:
        actual = stem + (f"_{seed}" if seed is not None else "")
        metadata = json.loads((Path("assets/rollouts") / (actual + ".json")).read_text())
        score = metadata.get("return_value", metadata.get("return"))
        prose += [
            f"![{description}, fitting seed {seed}](../assets/rollouts/{actual}.gif)",
            "",
            f"Fit {seed if seed is not None else 'reference'}: fixed episode return **{score:.3f}**, **{metadata['length']}** steps.",
            "",
        ]


def main():
    prose = [
        "# Visual simulation gallery",
        "",
        "Every new learned-policy GIF shows the predetermined first held-out reset300000 for one of fitting seeds0,1,2. All three fits are included. The episode return and length are verified against public numeric evidence. BFS receives a known model and is labeled separately. Failed episodes are retained; consult the three-seed table before judging a policy from one animation.",
        "",
        "## Navigation",
        "",
    ]
    for stem, description, seeds in [
        ("Maze_q_learning", "Owned stochastic maze — Q-learning", [0]),
        ("FrozenLake-v1_q_learning", "Official slippery FrozenLake — Q-learning", [0]),
        ("Taxi-v3_q_learning", "Official Taxi pickup and delivery — Q-learning", [0]),
        (
            "MiniGrid-FourRooms-v0_PPO_potential_curriculum",
            "Official Four Rooms — PPO with geometry features and a training-start curriculum",
            [0, 1, 2],
        ),
        (
            "MiniGrid-FourRooms-v0_PPO_potential_geometry",
            "Official Four Rooms — PPO with geometry features and potential shaping",
            [0, 1, 2],
        ),
        (
            "MiniGrid-DoorKey-6x6-v0_PPO_potential",
            "Official Door Key — fully observed PPO, potential-shaped training",
            [0, 1, 2],
        ),
        (
            "MiniGrid-FourRooms-v0_PPO_potential",
            "Four Rooms — flattened-map PPO, potential-shaped training",
            [0, 1, 2],
        ),
        ("MiniGrid-FourRooms-v0_PPO_sparse", "Four Rooms — PPO sparse-reward ablation", [0, 1, 2]),
        ("MiniGrid-DoorKey-6x6-v0_PPO_sparse", "Door Key — PPO sparse-reward ablation", [0, 1, 2]),
        ("MiniGrid-FourRooms-v0_BFS", "Four Rooms — known-model BFS planner", [None]),
        ("MiniGrid-DoorKey-6x6-v0_BFS", "Door Key — known-model BFS planner", [None]),
    ]:
        animation(prose, stem, description, seeds)
    prose += ["## Continuous control", ""]
    for stem, description, seeds in [
        ("LunarLander-v3_PPO", "LunarLander — PPO", [0]),
        ("HalfCheetah-v5_PPO_official", "Official MuJoCo HalfCheetah — PPO", [0, 1, 2]),
        ("HalfCheetah-v5_SAC_official", "Official MuJoCo HalfCheetah — SAC", [0, 1, 2]),
    ]:
        animation(prose, stem, description, seeds)
    rows = []
    paths = [
        p
        for family in ["simulations", "geometry", "curriculum"]
        for p in Path("results/reference", family).glob("MiniGrid*/*/metrics.json")
    ]
    for path in sorted(paths):
        m = json.loads(path.read_text())
        rows.append(
            dict(
                task=m["env"],
                profile=m["profile"],
                seed=m["seed"],
                success=m["success_fraction"],
                mean_return=m["test"]["mean"],
            )
        )
    frame = pd.DataFrame(rows)
    prose += [
        "## All maze fits",
        "",
        "| Task | Training profile | Seed | Test success | Mean official return |",
        "|---|---|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        prose.append(
            f"| {row.task} | {row.profile} | {row.seed} | {100 * row.success:.0f}% | {row.mean_return:.3f} |"
        )
    prose += [
        "",
        "Each rate uses100 predetermined test layouts. Full maps and map-derived shaping are explicit training choices, so these are not default partial-view MiniGrid results. See [protocol and primary references](simulations.md) and [all comparisons](results.md).",
    ]
    Path("docs/simulation_gallery.md").write_text("\n".join(prose) + "\n")


if __name__ == "__main__":
    main()
