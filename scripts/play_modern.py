"""Evaluate a published deterministic extension actor, with episodic memory."""

import argparse
from pathlib import Path

import numpy as np
import torch

from decision_lab.artifacts import sha256, write_json
from decision_lab.modern_deployment import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/deployment/modern.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    # The evaluator uses the official environment, but no trained SB3 model.
    from decision_lab.modern import evaluate

    actor = load(args.actor)
    evidence = evaluate(actor, actor.metadata["case"], np.arange(args.episodes) + 300000)
    write_json(args.output, dict(actor_sha256=sha256(args.actor), **evidence))
    print(float(evidence["returns"].mean()))


if __name__ == "__main__":
    main()
