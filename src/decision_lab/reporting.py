"""Rebuild figures and tables from complete numerical evidence, never from screenshots."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import write_json

EXPECTED = {
    "planning": 1,
    "tabular": 3,
    "neural": 12,
    "offline": 36,
    "control": 18,
    "approximation": 3,
    "lectures": 1,
}


def records(strict=True):
    collected = []
    counts = {key: 0 for key in EXPECTED}
    for path in sorted(Path("results/reference").rglob("run.json")):
        family = path.relative_to("results/reference").parts[0]
        if family not in EXPECTED:
            continue
        run = json.loads(path.read_text())
        if run["status"] != "complete":
            if strict:
                raise ValueError(f"Incomplete protocol: {path}")
            continue
        counts[family] += 1
        metrics = json.loads(path.with_name("metrics.json").read_text())
        collected.append((family, path.parent, run, metrics))
    if strict and counts != EXPECTED:
        raise ValueError(f"Reference counts differ: {counts}; expected {EXPECTED}")
    return collected, counts


def save_figure(fig, name):
    path = Path("assets/figures")
    path.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(path / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def lecture_figures():
    root = Path("assets/source_data/lectures")
    for task in ["stationary", "drifting"]:
        data = np.load(root / "bandits.npz")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        names = [
            "epsilon_average",
            "epsilon_constant",
            "ucb",
            "discounted_ucb",
            "gradient",
            "thompson",
        ]
        for name in names:
            rewards = data[f"{task}_{name}_rewards"]
            regret = data[f"{task}_{name}_regret"].cumsum(1)
            smoothed = (
                rewards[:, : rewards.shape[1] // 50 * 50].reshape(len(rewards), -1, 50).mean(2)
            )
            axes[0].plot(np.arange(smoothed.shape[1]) * 50 + 25, smoothed.mean(0), label=name)
            mean = regret.mean(0)
            se = regret.std(0, ddof=1) / np.sqrt(len(regret))
            x = np.arange(len(mean))
            axes[1].plot(x, mean, label=name)
            axes[1].fill_between(x, mean - 1.96 * se, mean + 1.96 * se, alpha=0.12)
        axes[0].set(
            xlabel="Interaction", ylabel="Reward (50-step mean)", title=f"{task.title()} bandits"
        )
        axes[1].set(
            xlabel="Interaction",
            ylabel="Cumulative pseudo-regret",
            title="Mean ± 1.96 SE across instances",
        )
        axes[1].legend(fontsize=8)
        save_figure(fig, f"bandits_{task}")
    data = np.load(root / "prediction.npz")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in data.files:
        y = data[name]
        mean = y.mean(0)
        se = y.std(0, ddof=1) / np.sqrt(len(y))
        x = np.arange(len(mean)) + 1
        ax.plot(x, mean, label=name)
        ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, alpha=0.12)
    ax.set(
        xlabel="Training episode",
        ylabel="RMSE against analytical values",
        title="Matched random-walk trajectories",
    )
    ax.legend()
    save_figure(fig, "prediction")
    data = np.load(root / "baird.npz")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in data.files:
        curve = data[name]
        for column, ax in enumerate(axes, 1):
            ax.semilogy(curve[:, 0], np.maximum(curve[:, column], 1e-16), label=name)
    axes[0].set(
        xlabel="Expected update", ylabel="Value RMS", title="Off-policy semigradient divergence"
    )
    axes[1].set(
        xlabel="Expected update", ylabel="MSPBE", title="Exact operators; zeros shown at 1e-16"
    )
    axes[1].legend()
    save_figure(fig, "off_policy_counterexample")


def maze_figure():
    from .deployment import load_policy
    from .toytext import MazeEnv

    env = MazeEnv()
    grid = np.array([[0.0 if cell == "#" else 1.0 for cell in row] for row in env.layout])
    methods = ["q_learning", "sarsa", "expected_sarsa", "double_q"]
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, method in zip(axes, methods):
        _, _, policy = load_policy(Path("assets/policies") / f"tabular_Maze_{method}_0.json")
        evidence = np.load(Path("assets/source_data/tabular") / f"Maze_{method}_0.npz")
        ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
        for state, (i, j) in enumerate(env.positions):
            if state == env.goal:
                continue
            di, dj = directions[policy[state]]
            ax.arrow(
                j,
                i,
                dj * 0.28,
                di * 0.28,
                head_width=0.14,
                color="#689fbb",
                length_includes_head=True,
            )
        trace = evidence["first_trace"]
        points = [env.positions[int(state)] for state in trace[:, 0]]
        if len(trace) and trace[-1, 2] == 1:
            points.append(env.positions[env.goal])
        points = np.array(points)
        ax.plot(points[:, 1], points[:, 0], color="#c54d16", linewidth=2, alpha=0.85)
        ax.scatter(*env.positions[env.start][::-1], color="#305ca8", s=35)
        ax.scatter(*env.positions[env.goal][::-1], color="#26934c", s=40)
        ax.set_title(
            method.replace("_", " ") + f"\nfirst test return {evidence['returns'][0]:.2f}",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Fitting seed 0: action maps and first 32 steps of fixed test reset 300000", fontsize=12
    )
    save_figure(fig, "maze_policies")
    env.close()


def report(strict=True):
    all_records, counts = records(strict)
    rows = []
    histories = []
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for family, directory, run, metrics in all_records:
        for row in metrics.get("rows", [metrics]):
            if family == "tabular":
                task = row["env"]
                method = row["method"]
            elif family == "neural":
                task = "Wireless online"
                method = row["recipe"]
            elif family == "offline":
                task = f"Wireless offline epsilon={row['behavior_epsilon']:g}"
                method = row["recipe"]
            elif family == "control":
                task = row["env"]
                method = row["algorithm"]
            elif family == "approximation":
                task = "MountainCar-v0"
                method = f"SARSA(lambda={row['lambda']:g})"
            else:
                continue
            rows.append(
                {
                    "task": task,
                    "method": method,
                    "seed": row["seed"],
                    "test_return": row["test"]["mean"],
                    "episode_bootstrap_low": row["test"]["episode_bootstrap_ci95"][0],
                    "episode_bootstrap_high": row["test"]["episode_bootstrap_ci95"][1],
                    "exact_normalized_gap": row.get("normalized_exact_policy_gap"),
                    "shift_return": row.get("shift_test", row.get("shift", {})).get("mean")
                    if row.get("shift_test", row.get("shift"))
                    else None,
                    "source_metrics": str(directory / "metrics.json"),
                }
            )
            history = row.get("validation_history")
            if history:
                histories.append((family, task, method, row["seed"], history))
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["task", "method"], sort=False)
        .agg(
            n_fits=("seed", "size"),
            mean_return=("test_return", "mean"),
            fit_sd=("test_return", "std"),
            worst_fit_return=("test_return", "min"),
            best_fit_return=("test_return", "max"),
            mean_exact_gap=("exact_normalized_gap", "mean"),
            mean_shift_return=("shift_return", "mean"),
        )
        .reset_index()
    )
    Path("docs").mkdir(exist_ok=True)
    frame.to_csv("docs/individual_fits.csv", index=False)
    summary.to_csv("docs/comparison.csv", index=False)
    write_json(
        "docs/experiment_index.json",
        {
            "counts": counts,
            "total_protocols": sum(counts.values()),
            "protocols": [
                {"family": family, "path": str(path), "elapsed_seconds": run["elapsed_seconds"]}
                for family, path, run, _ in all_records
            ],
        },
    )
    for task in summary.task.unique():
        subset = summary[summary.task == task]
        fig, ax = plt.subplots(figsize=(max(6, len(subset) * 1.4), 4))
        x = np.arange(len(subset))
        ax.bar(x, subset.mean_return, color="#3579a9", alpha=0.65)
        for i, method in enumerate(subset.method):
            individual = frame[(frame.task == task) & (frame.method == method)]
            ax.scatter(
                np.full(len(individual), i), individual.test_return, color="black", s=24, zorder=3
            )
        ax.set_xticks(x, subset.method, rotation=20, ha="right")
        ax.set(ylabel="Held-out episode return", title=f"{task}: mean bar, individual fit dots")
        save_figure(fig, "comparison_" + task.replace(" ", "_").replace("=", "").replace("/", "_"))
    for task in sorted(set(item[1] for item in histories)):
        fig, ax = plt.subplots(figsize=(8, 4))
        methods = list(dict.fromkeys(item[2] for item in histories if item[1] == task))
        colors = plt.cm.tab10(np.arange(len(methods)))
        labelled = set()
        for family, t, method, seed, history in histories:
            if t != task:
                continue
            xs = [row.get("step", row.get("steps")) for row in history]
            ys = [
                row.get(
                    "discounted_mean",
                    row.get(
                        "mean_return",
                        row.get("return", row.get("mean_validation_return", row.get("score"))),
                    ),
                )
                for row in history
            ]
            label = method if method not in labelled else None
            labelled.add(method)
            ax.plot(xs, ys, color=colors[methods.index(method)], alpha=0.65, label=label)
        ax.set(
            xlabel="Training interaction",
            ylabel="Validation return",
            title=f"{task}: every fitting seed",
        )
        ax.legend(fontsize=8)
        save_figure(fig, "learning_" + task.replace(" ", "_").replace("/", "_"))
    planning = next(
        (metrics for family, _, _, metrics in all_records if family == "planning"), None
    )
    if planning:
        table = pd.DataFrame(planning["pareto"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for profile, ax in zip(["per_user", "global"], axes):
            selected = table[table.profile == profile]
            for priorities in [[1.0, 1.0], [1.0, 3.0]]:
                for fair in [False, True]:
                    subset = selected[
                        (selected.priorities.apply(lambda p: p == priorities))
                        & (selected.fairness_constraint == fair)
                    ]
                    ax.plot(
                        subset.normalized_energy,
                        subset.normalized_weighted_loss,
                        "o-",
                        label=f"weights {priorities}; "
                        + ("user 1 service ≥ .12" if fair else "no service floor"),
                    )
            ax.set(
                xlabel="Normalized discounted energy",
                ylabel="Normalized weighted packet loss",
                title="Per-user channel (324 states)"
                if profile == "per_user"
                else "Global channel (108 states)",
            )
            ax.legend(fontsize=7)
        save_figure(fig, "constrained_planning")
        data = np.load("assets/source_data/planning.npz")
        fig, ax = plt.subplots(figsize=(7, 4))
        for profile in ["per_user", "global"]:
            ax.semilogy(data[f"{profile}_VI_residual"], label=profile)
        ax.set(
            xlabel="VI sweep",
            ylabel="Bellman residual",
            title="Discount .99: certified contraction convergence",
        )
        ax.legend()
        save_figure(fig, "planning_convergence")
    offline = summary[summary.task.str.contains("offline")]
    if len(offline):
        pivot = offline.pivot(index="task", columns="method", values="mean_exact_gap")
        fig, ax = plt.subplots(figsize=(8, 4))
        artist = ax.imshow(pivot.values, cmap="magma")
        ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", color="white")
        ax.set_title("Mean normalized exact deployment gap (three fits)")
        fig.colorbar(artist, ax=ax)
        save_figure(fig, "offline_coverage")
    if Path("assets/policies/tabular_Maze_q_learning_0.json").exists():
        maze_figure()
    if Path("assets/source_data/lectures/bandits.npz").exists():
        lecture_figures()
    prose = [
        "# Measured comparisons",
        "",
        f"{sum(counts.values())} completed reference protocols. Every fit is listed in [individual_fits.csv](individual_fits.csv); [comparison.csv](comparison.csv) aggregates the fitting seeds.",
        "",
        "Bars show the mean of fitting-seed test means; dots show every fit. Fit SD measures training variability across only three fits. Episode bootstrap intervals in the individual table quantify resets conditional on one fitted policy; they do not measure training uncertainty. No test return selects a checkpoint or hyperparameter.",
        "",
        "| Task | Method | Fits | Mean test return | Fit SD | Worst fit | Mean exact gap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        gap = "—" if pd.isna(row.mean_exact_gap) else f"{row.mean_exact_gap:.5f}"
        prose.append(
            f"| {row.task} | {row.method} | {row.n_fits} | {row.mean_return:.2f} | {row.fit_sd:.2f} | {row.worst_fit_return:.2f} | {gap} |"
        )
    prose += [
        "",
        "Exact normalized gaps use the full discounted MDP from the declared initial distribution. They measure a continuing/model-defined objective, while time-limited test episode returns measure finite rollouts. Wireless offline priorities [1,3] differ from online [1,1], so their raw returns must not be compared across those tasks.",
        "",
        "Training budgets differ across PPO/DQN/SAC control cases and are recorded in configs and run records. These compare practical preset protocols, not equal-compute algorithm rankings. Planning methods receive the exact transition model; learned methods receive sampled transitions.",
        "",
        "CQL/IQL are discrete adaptations on a small known-action-mask wireless environment, not reproductions of the papers’ D4RL benchmark claims. Limited behavior coverage can make conservative policy extraction retain the biased behavior. Quantile crossing is reported for QR-DQN; return quantiles do not certify epistemic uncertainty.",
        "",
        "Plots rebuild from numeric NPZ arrays and JSON histories using `decision-lab report`; every plot has PNG and PDF versions. Scientific figures use fixed test protocols, including poor fitted seeds.",
    ]
    Path("docs/results.md").write_text("\n".join(prose) + "\n")
    print("Reference protocols:", counts, "Fitted policies:", len(frame))
