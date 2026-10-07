"""Full-budget planning and tabular comparisons; procedural checks use separate paths."""

from pathlib import Path

import numpy as np

from .artifacts import Run, interval, sha256
from .mdp import (
    average_occupancy,
    bellman_lp,
    discounted_occupancy,
    evaluate_policy,
    finite_horizon,
    policy_iteration,
    value_iteration,
)
from .tabular import learn, rollout_policy
from .toytext import factory, model_from_env
from .wireless import WirelessEnv, WirelessModel


def planning(config, output="results/reference/planning", quick=False):
    root = Path(__file__).parent
    run = Run(output, config | {"quick": quick}, [__file__, root / "mdp.py", root / "wireless.py"])
    rows = []
    pareto = []
    data = {}
    for profile in config["profiles"]:
        model = WirelessModel(config["p_data"], config["p_energy"], profile)
        mdp = model.finite_mdp()
        gamma = config["gamma"]
        vi = value_iteration(mdp, gamma)
        pi = policy_iteration(mdp, gamma)
        primal = bellman_lp(mdp, gamma)
        dual = discounted_occupancy(mdp, gamma)
        avg = average_occupancy(mdp)
        vh, ph = finite_horizon(mdp, config["horizon"])
        for name, result in [("VI", vi), ("PI", pi), ("occupancy", dual)]:
            data[f"{profile}_{name}_value"] = result["value"]
            data[f"{profile}_{name}_policy"] = result["policy"]
        data[f"{profile}_VI_residual"] = vi["history"]
        data[f"{profile}_PI_residual"] = pi["history"]
        data[f"{profile}_finite_value"] = vh
        data[f"{profile}_finite_policy"] = ph
        data[f"{profile}_average_policy"] = avg["policy"]
        data[f"{profile}_average_bias"] = avg["bias"]
        rows.append(
            {
                "profile": profile,
                "states": mdp.states,
                "VI_iterations": vi["iterations"],
                "PI_iterations": pi["iterations"],
                "VI_value_error_bound": vi["error_bound"],
                "normalized_optimal_loss": float(-(1 - gamma) * vi["value"].mean()),
                "VI_PI_max_value_difference": float(np.max(abs(vi["value"] - pi["value"]))),
                "VI_LP_max_value_difference": float(np.max(abs(vi["value"] - primal["value"]))),
                "occupancy_flow_residual": dual["flow_residual"],
                "discounted_primal_dual_gap": abs(primal["objective"] - dual["dual_objective"]),
                "average_packet_loss": -avg["average_reward"],
                "average_primal_dual_gap": abs(avg["average_reward"] - avg["gain_dual"]),
                "average_stationarity_residual": avg["stationarity_residual"],
                "steady_energy_lower_bound_loss": 2 * config["p_data"] - 2 * config["p_energy"],
            }
        )
        for priorities in [(1.0, 1.0), tuple(config["priority_extension"])]:
            weighted = WirelessModel(config["p_data"], config["p_energy"], profile, priorities)
            wm = weighted.finite_mdp()
            for budget in config["energy_budgets"]:
                for fair in [False, True]:
                    costs = weighted.energy_cost[None]
                    caps = [budget]
                    if fair:
                        costs = np.concatenate([costs, -weighted.user_service[0:1]])
                        caps.append(-config["minimum_user1_service"])
                    result = discounted_occupancy(wm, gamma, costs=costs, budgets=caps)
                    service = (1 - gamma) * (weighted.user_service * result["occupancy"]).sum(
                        (1, 2)
                    )
                    pareto.append(
                        {
                            "profile": profile,
                            "priorities": list(priorities),
                            "fairness_constraint": fair,
                            "energy_budget": budget,
                            "normalized_weighted_loss": -(1 - gamma) * result["objective"],
                            "normalized_energy": float(result["normalized_costs"][0]),
                            "user_service": service.tolist(),
                            "flow_residual": result["flow_residual"],
                            "budget_shadow_prices": result["budget_shadow_prices"].tolist(),
                            "primal_dual_gap": abs(result["objective"] - result["dual_objective"]),
                        }
                    )
    target = run.directory / "source_data" if quick else Path("assets/source_data")
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "planning.npz", **data)
    run.finish(
        {
            "rows": rows,
            "pareto": pareto,
            "protocol": "Exact model-aware planning. Uniform initial distribution for discounted values. Average reward is justified only for the wireless interior-arrival unichain model; not a universal multichain guarantee. Equal-priority energy/loss tradeoff is linear by packet conservation. Priority and service-floor extensions are explicitly separate objectives.",
        }
    )


def tabular(config, seed, output="results/reference/tabular", quick=False):
    root = Path(__file__).parent
    run = Run(
        Path(output) / f"seed{seed}",
        config | {"seed": seed, "quick": quick},
        [__file__, root / "tabular.py", root / "toytext.py", root / "wireless.py", root / "mdp.py"],
    )
    target = run.directory / "source_data" if quick else Path("assets/source_data/tabular")
    target.mkdir(parents=True, exist_ok=True)
    checkpoints = run.directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    rows = []
    for name, steps in config["budgets"].items():
        wireless = name == "Wireless"
        if wireless:
            model = WirelessModel()

            def make():
                return WirelessEnv(model)

            mdp = model.finite_mdp()
            mu = np.ones(mdp.states) / mdp.states
        else:
            make = factory(name)
            env = make()
            mdp = model_from_env(env)
            mu = getattr(env.unwrapped, "initial_state_distrib", None)
            if mu is None:
                obs, info = env.reset(seed=0)
                mu = np.zeros(mdp.states)
                mu[int(obs)] = 1
            env.close()
        optimum = value_iteration(mdp, config["gamma"])
        for method in config["methods"]:
            fitted = learn(
                make,
                mdp.states,
                mdp.actions,
                steps,
                method,
                seed,
                config["gamma"],
                config["alpha"],
                config["validation_every"],
                config["validation_episodes"],
                wireless,
            )
            policy = np.where(mdp.legal, fitted["q"], -np.inf).argmax(1)
            evidence = rollout_policy(
                make, policy, np.arange(config["test_episodes"]) + 300000, config["gamma"]
            )
            value = evaluate_policy(mdp, policy, config["gamma"])
            stem = f"{name}_{method}_{seed}"
            np.savez_compressed(
                target / f"{stem}.npz",
                **evidence,
                training_episodes=fitted["training_episodes"],
                state_action_visits=fitted["counts"],
            )
            checkpoint = checkpoints / f"{stem}.npz"
            np.savez_compressed(checkpoint, q=fitted["q"], policy=policy)
            row = {
                "env": name,
                "method": method,
                "seed": seed,
                "steps": steps,
                "selected_step": fitted["selected_step"],
                "best_validation_return": fitted["best_validation_return"],
                "test": interval(evidence["returns"], seed),
                "discounted_test": interval(evidence["discounted_returns"], seed),
                "exact_initial_discounted_value": float(mu @ value),
                "exact_initial_optimal_value": float(mu @ optimum["value"]),
                "normalized_exact_policy_gap": float(
                    (1 - config["gamma"]) * (mu @ (optimum["value"] - value))
                ),
                "test_termination_fraction": float(evidence["terminated"].mean()),
                "test_truncation_fraction": float(evidence["truncated"].mean()),
                "visited_legal_fraction": float((fitted["counts"][mdp.legal] > 0).mean()),
                "validation_history": fitted["validation_history"],
                "checkpoint_sha256": sha256(checkpoint),
            }
            if wireless:
                row["test_packet_loss_fraction"] = float(
                    evidence["expired"].sum() / evidence["offered"].sum()
                )
            rows.append(row)
            print("Complete tabular", name, method, seed, row["test"]["mean"], flush=True)
    run.finish(
        {
            "rows": rows,
            "protocol": "Four owned tabular algorithms, matched environment-interaction budgets, three fitting seeds, fixed independent validation/test resets, legal masks and correct time-limit bootstrap. Exact discounted policy gaps are post-fit model-aware diagnostics; Gym time-limited raw returns describe a different finite-rollout quantity. No test-based checkpoint selection.",
        }
    )
