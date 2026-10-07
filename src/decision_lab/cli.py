"""Command line entry point; optional deep-learning dependencies load on demand."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import write_json


def main():
    parser = argparse.ArgumentParser(
        description="Reproducible sequential decision experiments and deployment"
    )
    parser.add_argument(
        "command",
        choices=[
            "planning",
            "tabular",
            "neural",
            "offline",
            "control",
            "simulations",
            "geometry",
            "curriculum",
            "modern",
            "lectures",
            "approximation",
            "rollout",
            "report",
            "audit",
        ],
    )
    parser.add_argument("--config")
    parser.add_argument(
        "--case", help="Filter simulation cases by environment, algorithm or profile substring"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--policy")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.command == "rollout":
        from .deployment import load_policy
        from .tabular import rollout_policy

        if not args.policy:
            parser.error("rollout requires --policy")
        _, make, policy = load_policy(args.policy)
        result = rollout_policy(make, policy, np.arange(args.episodes) + 300000)
        write_json(args.output or "results/deployment/rollout.json", result)
        print(float(result["returns"].mean()))
        return
    if args.command == "report":
        from .reporting import report

        report()
        return
    if args.command == "audit":
        from .deployment import export_policies

        print("Audited finite policy exports:", len(export_policies()))
        return
    config_path = Path(args.config or f"configs/{args.command}.json")
    if args.config is None and not config_path.exists():
        from importlib.resources import files

        config = json.loads(
            (files("decision_lab") / "default_configs" / f"{args.command}.json").read_text()
        )
    else:
        config = json.loads(config_path.read_text())
    output = args.output or f"results/{'smoke' if args.quick else 'reference'}/{args.command}"
    if args.command == "planning":
        from .experiments import planning

        planning(config, output, args.quick)
    elif args.command == "tabular":
        from .experiments import tabular

        if args.quick:
            config |= {
                "budgets": {key: 1000 for key in config["budgets"]},
                "validation_episodes": 2,
                "test_episodes": 3,
            }
        tabular(config, args.seed, output, args.quick)
    elif args.command in ["neural", "offline"]:
        if args.command == "neural":
            from .deep_wireless import fit

            if args.quick:
                config |= {
                    "steps": 2048,
                    "validation_every": 2048,
                    "validation_episodes": 2,
                    "test_episodes": 3,
                }
        else:
            from .offline import fit

            if args.quick:
                config |= {
                    "dataset_steps": 1000,
                    "updates": 20,
                    "test_episodes": 3,
                    "behavior_epsilon": [0.05],
                }
        from .protocols import completed_batch

        if completed_batch(args.command, config, args.seed, args.device, args.quick, output):
            print("Already complete; preserved numeric evidence")
            return
        fit(config, args.seed, args.device, args.quick, output)
    elif args.command == "control":
        from .control import train_case

        if args.quick:
            config |= {
                "validation_episodes": 2,
                "test_episodes": 3,
                "shift_episodes": 2,
                "evaluation_every": 2048,
            }
        for case in config["cases"]:
            if args.quick:
                case |= {"steps": 2048}
            train_case(config, case, args.seed, output, args.quick)
    elif args.command in ["simulations", "geometry", "curriculum"]:
        from .simulations import identifier

        if args.command == "curriculum":
            from .curriculum import train_case
        elif args.command == "geometry":
            from .geometry import train_case
        else:
            from .simulations import train_case

        config["device"] = args.device
        if args.quick:
            config |= {"validation_episodes": 2, "test_episodes": 3, "evaluation_every": 2048}
        matched = [
            case for case in config["cases"] if not args.case or args.case in identifier(case)
        ]
        if not matched:
            parser.error("--case matches no simulation case")
        for case in matched:
            evidence = (
                Path("assets/source_data/simulations") / f"{identifier(case)}_{args.seed}.npz"
            )
            canonical = Path("results/reference") / args.command
            if (
                not args.quick
                and evidence.exists()
                and Path(output).resolve() != canonical.resolve()
            ):
                parser.error(
                    "This --output would overwrite published episode evidence. "
                    "Use a distinct case profile or a separate checkout for changed full protocols."
                )
            if args.quick:
                case |= {"steps": 2048}
            train_case(config, case, args.seed, output, args.quick)
    elif args.command == "modern":
        from .modern import identifier, train

        config["device"] = args.device
        if args.quick:
            config |= {"validation_episodes": 2, "test_episodes": 3, "evaluation_every": 2048}
        matched = [
            case for case in config["cases"] if not args.case or args.case in identifier(case)
        ]
        if not matched:
            parser.error("--case matches no modern case")
        for case in matched:
            if args.quick:
                case = case | {"steps": 2048}
            if (
                not args.quick
                and Path(output).resolve() != Path("results/reference/modern").resolve()
            ):
                evidence = Path("assets/source_data/modern") / f"{identifier(case)}_{args.seed}.npz"
                if evidence.exists():
                    parser.error(
                        "Use a separate checkout to preserve published evidence when changing full protocols"
                    )
            train(config, case, args.seed, output, args.quick)
    elif args.command == "lectures":
        from .lecture_studies import run

        if args.quick:
            config = {
                "baird_steps": 1000,
                "prediction_seeds": 3,
                "prediction_episodes": 10,
                "bandit_runs": 3,
                "bandit_steps": 20,
            }
        run(config, output, args.quick)
    else:
        from .approximation import fit

        if args.quick:
            config |= {
                "episodes": 2,
                "validation_every": 2,
                "validation_episodes": 2,
                "test_episodes": 3,
            }
        fit(config, args.seed, output, args.quick)


if __name__ == "__main__":
    main()
