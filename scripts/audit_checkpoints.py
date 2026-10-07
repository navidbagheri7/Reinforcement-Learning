"""Private-weight deployment audit; public release validation is independent."""

from pathlib import Path

import numpy as np
import torch

from decision_lab.approximation import Tiles, rollout
from decision_lab.artifacts import sha256, write_json
from decision_lab.control import ALGORITHMS, evaluate, shift_spec
from decision_lab.deployment import export_policies
from decision_lab.reporting import records


def main():
    protocols, _ = records()
    torch.set_num_threads(1)
    audits = []
    for family, path, run, metrics in protocols:
        if family == "control":
            checkpoint = path / "checkpoints/best_model.zip"
            assert sha256(checkpoint) == metrics["checkpoint_sha256"]
            model = ALGORITHMS[metrics["algorithm"]].load(checkpoint, device="cpu")
            reference = np.load(
                Path("assets/source_data/control")
                / f"{metrics['env']}_{metrics['algorithm']}_{metrics['seed']}.npz"
            )
            for label, kwargs in [("test", None), ("shift", shift_spec(metrics["env"]))]:
                if label == "shift" and kwargs is None:
                    continue
                evidence = evaluate(model, metrics["env"], reference[f"{label}_seeds"][:3], kwargs)
                for key in [
                    "returns",
                    "lengths",
                    "terminated",
                    "truncated",
                    "first_observations",
                    "first_actions",
                ]:
                    expected = (
                        reference[f"{label}_{key}"]
                        if key.startswith("first_")
                        else reference[f"{label}_{key}"][:3]
                    )
                    assert np.array_equal(evidence[key], expected), (path, label, key)
            audits.append(
                {
                    "path": str(checkpoint),
                    "sha256": sha256(checkpoint),
                    "fresh_test_episodes": 3,
                    "fresh_shift_episodes": 3 if shift_spec(metrics["env"]) else 0,
                }
            )
        elif family == "approximation":
            for row in metrics["rows"]:
                name = f"lambda{row['lambda']:g}_{row['seed']}"
                checkpoint = path / "checkpoints" / f"{name}.npz"
                assert sha256(checkpoint) == row["checkpoint_sha256"]
                weights = np.load(checkpoint)
                tiles = Tiles(int(weights["tilings"]), int(weights["bins"]))
                evidence = rollout(weights["weights"], tiles, np.arange(3) + 300000)
                reference = np.load(Path("assets/source_data/approximation") / f"{name}.npz")
                for key in ["returns", "lengths", "success", "first_trace"]:
                    expected = reference[key] if key == "first_trace" else reference[key][:3]
                    assert np.array_equal(evidence[key], expected), (name, key)
                public = Path("assets/linear_policies") / f"{name}.npz"
                public.parent.mkdir(parents=True, exist_ok=True)
                if not public.exists():
                    public.write_bytes(checkpoint.read_bytes())
                assert sha256(public) == sha256(checkpoint)
                audits.append(
                    {
                        "path": str(checkpoint),
                        "sha256": sha256(checkpoint),
                        "fresh_test_episodes": 3,
                        "public_weights": str(public),
                    }
                )
    finite = export_policies()
    write_json(
        "docs/checkpoint_audit.json",
        {"status": "passed", "finite_exports": len(finite), "other_policies": audits},
    )
    print(
        "Audited",
        len(finite),
        "finite policies and",
        len(audits),
        "continuous-observation checkpoints",
    )


if __name__ == "__main__":
    main()
