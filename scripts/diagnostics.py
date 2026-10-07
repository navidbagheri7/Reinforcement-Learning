"""Supplementary numerical checks computed from preserved fitted evidence."""

import json
from pathlib import Path

import numpy as np

from decision_lab.artifacts import write_json
from decision_lab.mdp import evaluate_policy
from decision_lab.wireless import WirelessModel


def main():
    model = WirelessModel()
    mdp = model.finite_mdp()
    rows = []
    value_rows = []
    for file in sorted(Path("assets/source_data/neural").glob("*.npz")):
        with np.load(file, allow_pickle=False) as data:
            raw = data["q"]
            scores = raw.mean(-1) if raw.ndim == 3 else raw
            policy = data["policy"]
            predicted = scores[np.arange(len(policy)), policy]
            exact = evaluate_policy(mdp, policy, 0.99)
            residual = mdp.q(predicted, 0.99).max(1) - predicted
            value_rows.append(
                {
                    "source": str(file),
                    "mean_predicted_value": float(predicted.mean()),
                    "mean_exact_policy_value": float(exact.mean()),
                    "max_value_error": float(np.max(abs(predicted - exact))),
                    "max_optimal_bellman_residual": float(np.max(abs(residual))),
                }
            )
    for file in sorted(Path("assets/source_data/neural").glob("qr_dqn_*.npz")):
        with np.load(file, allow_pickle=False) as data:
            q = data["q"]
            legal = q[model.legal]
            rows.append(
                {
                    "source": str(file),
                    "all_heads_crossing_fraction": float((np.diff(q, axis=-1) < 0).mean()),
                    "legal_heads_crossing_fraction": float((np.diff(legal, axis=-1) < 0).mean()),
                    "learned_legal_quantile_output_range": [float(legal.min()), float(legal.max())],
                    "note": "Illegal action heads are not trained for deployment. Crossing diagnostics neither sort targets nor imply epistemic calibration.",
                }
            )
    write_json(
        "docs/diagnostics.json",
        {
            "qr_dqn": rows,
            "neural_value": value_rows,
            "scope": "Post-fit model-aware diagnostics; no checkpoint selection. Early validation-best policies can perform well before their critics calibrate to the long discounted horizon.",
        },
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
