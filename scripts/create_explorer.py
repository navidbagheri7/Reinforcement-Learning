"""Build a concept-organized, offline explorer from the measured RL evidence."""

import json
from pathlib import Path

import numpy as np

from decision_lab.artifacts import sha256, write_json
from decision_lab.toytext import MazeEnv, model_from_env


def main():
    rows, bindings = [], {}
    families = [
        "tabular",
        "neural",
        "offline",
        "control",
        "approximation",
        "simulations",
        "geometry",
        "curriculum",
        "modern",
    ]
    for family in families:
        for path in sorted(Path("results/reference", family).rglob("metrics.json")):
            bindings[str(path)] = sha256(path)
            metrics = json.loads(path.read_text())
            for row in metrics.get("rows", [metrics]):
                if family == "tabular":
                    task, method = row["env"], row["method"]
                    source = (
                        Path("assets/source_data/tabular") / f"{task}_{method}_{row['seed']}.npz"
                    )
                    curve = [
                        (p["step"], p["mean_validation_return"]) for p in row["validation_history"]
                    ]
                elif family == "neural":
                    task, method = "Wireless online", row["recipe"]
                    source = Path("assets/source_data/neural") / f"{method}_{row['seed']}.npz"
                    curve = [(p["step"], p["discounted_mean"]) for p in row["validation_history"]]
                elif family == "offline":
                    task, method = (
                        f"Wireless offline · behavior epsilon {row['behavior_epsilon']:g}",
                        row["recipe"],
                    )
                    source = (
                        Path("assets/source_data/offline")
                        / f"epsilon{row['behavior_epsilon']:g}_{method}_{row['seed']}.npz"
                    )
                    curve = np.load(source, allow_pickle=False)["losses"].tolist()
                elif family == "approximation":
                    task, method = "MountainCar-v0", f"SARSA lambda {row['lambda']:g}"
                    source = (
                        Path("assets/source_data/approximation")
                        / f"lambda{row['lambda']:g}_{row['seed']}.npz"
                    )
                    curve = [(p["steps"], p["return"]) for p in row["validation_history"]]
                else:
                    task, method = (
                        row["env"],
                        row["algorithm"] + (" · " + row["profile"] if "profile" in row else ""),
                    )
                    stem = (
                        f"{row['env']}_{row['algorithm']}"
                        + ("_" + row["profile"] if "profile" in row else "")
                        + f"_{row['seed']}"
                    )
                    source = Path(
                        "assets/source_data/control"
                        if family == "control"
                        else "assets/source_data/modern"
                        if family == "modern"
                        else "assets/source_data/simulations"
                    ) / (stem + ".npz")
                    curve = [(p["step"], p["mean_return"]) for p in row["validation_history"]]
                with np.load(source, allow_pickle=False) as data:
                    returns = data["test_returns" if family == "control" else "returns"]
                np.testing.assert_allclose(returns.mean(), row["test"]["mean"], rtol=0, atol=1e-10)
                bindings[str(source)] = sha256(source)
                group = (
                    "Simulations and memory"
                    if family in {"simulations", "geometry", "curriculum", "modern"}
                    else {
                        "tabular": "Tabular control",
                        "neural": "Deep and distributional control",
                        "offline": "Offline learning",
                        "control": "Standard control",
                        "approximation": "Linear approximation",
                    }[family]
                )
                gif = None
                if family in {"simulations", "geometry", "curriculum", "modern", "control"}:
                    candidate = Path("assets/rollouts") / (source.stem + ".gif")
                    if candidate.exists():
                        gif = "../" + str(candidate)
                if family == "tabular":
                    candidate = (
                        Path("assets/rollouts") / f"{row['env']}_{row['method']}_{row['seed']}.gif"
                    )
                    if candidate.exists():
                        gif = "../" + str(candidate)
                if gif:
                    bindings[gif[3:]] = sha256(gif[3:])
                rows.append(
                    dict(
                        group=group,
                        task=task,
                        method=method,
                        seed=row["seed"],
                        instances=1,
                        score=row["test"]["mean"],
                        ci=row["test"]["episode_bootstrap_ci95"],
                        selected=row.get("selected_step", row.get("selected_episode")),
                        selection_unit="episode" if family == "approximation" else "interaction",
                        gap=row.get("normalized_exact_policy_gap"),
                        coverage=row.get("visited_legal_fraction"),
                        shift=row.get("shift_test", row.get("shift")),
                        success=row.get("success_fraction", row.get("test_success_fraction")),
                        curve=curve,
                        curve_label="Optimizer loss (objectives differ across algorithms)"
                        if family == "offline"
                        else "Validation return (discounted for wireless online)",
                        curve_x="Gradient update"
                        if family == "offline"
                        else "Training interactions",
                        returns=returns.tolist(),
                        metrics="../" + str(path),
                        source="../" + str(source),
                        gif=gif,
                        profile=row.get(
                            "profile", "finite" if family == "tabular" else "declared preset"
                        ),
                    )
                )
    planning_path = Path("results/reference/planning/metrics.json")
    planning = json.loads(planning_path.read_text())
    bindings[str(planning_path)] = sha256(planning_path)
    lecture_path = Path("results/reference/lectures/metrics.json")
    lectures = json.loads(lecture_path.read_text())
    bindings[str(lecture_path)] = sha256(lecture_path)
    for name in ["bandits", "prediction", "baird"]:
        source = Path("assets/source_data/lectures") / (name + ".npz")
        bindings[str(source)] = sha256(source)
        with np.load(source, allow_pickle=False) as data:
            for key in data.files:
                value = data[key]
                if name == "bandits" and not key.endswith("_regret"):
                    continue
                if name == "baird":
                    curve = value[:, [0, 1]].tolist()
                    final = float(value[-1, 1])
                else:
                    values = value.cumsum(1) if name == "bandits" else value
                    indexes = np.unique(
                        np.linspace(0, values.shape[1] - 1, min(200, values.shape[1])).astype(int)
                    )
                    curve = np.column_stack([indexes + 1, values.mean(0)[indexes]]).tolist()
                    final = float(values[:, -1].mean())
                rows.append(
                    dict(
                        group="Prediction and bandits",
                        task=name,
                        method=key.removesuffix("_regret"),
                        seed="all",
                        instances=1 if name == "baird" else value.shape[0],
                        score=final,
                        ci=None,
                        selected=None,
                        gap=None,
                        coverage=None,
                        shift=None,
                        success=None,
                        curve=curve,
                        curve_label="Mean cumulative pseudo-regret"
                        if name == "bandits"
                        else "Value RMS"
                        if name == "baird"
                        else "Mean prediction RMSE",
                        curve_x="Expected update"
                        if name == "baird"
                        else "Interaction"
                        if name == "bandits"
                        else "Training episode",
                        returns=[],
                        metrics="../" + str(lecture_path),
                        source="../" + str(source),
                        gif=None,
                        profile="Expected known-model operator"
                        if name == "baird"
                        else "Independent instances; shared trajectories/noise across methods",
                    )
                )
    env = MazeEnv()
    mdp = model_from_env(env)
    transitions = [
        [
            [
                [int(j), float(p), float(mdp.continuation[s, a, j])]
                for j, p in enumerate(mdp.transition[s, a])
                if p > 0
            ]
            for a in range(mdp.actions)
        ]
        for s in range(mdp.states)
    ]
    maze = dict(
        layout=env.layout,
        positions=env.positions,
        start=env.start,
        goal=env.goal,
        reward=mdp.reward.tolist(),
        transitions=transitions,
        legal=mdp.legal.tolist(),
    )
    env.close()
    evidence = dict(rows=rows, planning=planning, lectures=lectures, maze=maze)
    template = Path("scripts/templates/rl_explorer.html")
    script = Path("scripts/templates/rl_explorer.js")
    output = Path("docs/explorer.html")
    encoded = json.dumps(evidence, allow_nan=False).replace("<", "\\u003c")
    output.write_text(
        template.read_text()
        .replace("__EVIDENCE__", encoded)
        .replace("__SCRIPT__", script.read_text())
    )
    for path in [
        template,
        script,
        Path(__file__),
        Path("src/decision_lab/toytext.py"),
        Path("src/decision_lab/mdp.py"),
    ]:
        bindings[str(path)] = sha256(path)
    write_json(
        "docs/explorer_manifest.json",
        dict(
            status="complete"
            if len(list(Path("results/reference/modern").glob("*/*/metrics.json"))) == 18
            else "partial",
            html_sha256=sha256(output),
            bindings=bindings,
            rows=len(rows),
            groups=8,
            protocol="Measured presets retain every fitting seed. Bandit/prediction curves use at most 200 evenly spaced recorded mean values; no interpolation of intermediate policies. Live maze sweeps solve the declared known model, not recorded trained policies.",
        ),
    )
    print("Embedded", len(rows), "measured presets and live maze planning")


if __name__ == "__main__":
    main()
