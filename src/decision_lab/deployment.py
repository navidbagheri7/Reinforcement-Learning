"""Versioned finite-policy exports and fresh held-out checkpoint checks."""

import json
from pathlib import Path

import numpy as np

from .artifacts import sha256, write_json
from .tabular import rollout_policy
from .toytext import factory
from .wireless import WirelessEnv, WirelessModel


def load_policy(path):
    record = json.loads(Path(path).read_text())
    if record.get("schema_version") != 1:
        raise ValueError("Unsupported policy schema")
    if record["env"] == "Wireless":
        model = WirelessModel(**record["wireless"])

        def make():
            return WirelessEnv(model)

        states = len(model.states)
        legal = model.legal
    else:
        make = factory(record["env"])
        env = make()
        states = env.observation_space.n
        from .toytext import model_from_env

        legal = model_from_env(env).legal
        env.close()
    policy = np.asarray(record["policy"])
    if policy.shape != (states,) or not np.issubdtype(policy.dtype, np.integer):
        raise ValueError("Invalid finite policy shape/type")
    if (
        np.any((policy < 0) | (policy >= legal.shape[1]))
        or not legal[np.arange(states), policy].all()
    ):
        raise ValueError("Illegal finite policy action")
    return record, make, policy


def export_policies():
    destination = Path("assets/policies")
    destination.mkdir(parents=True, exist_ok=True)
    exported = []
    for path in sorted(Path("results/reference").rglob("metrics.json")):
        relative = path.relative_to("results/reference")
        if relative.parts[0] not in ["tabular", "neural", "offline"]:
            continue
        metrics = json.loads(path.read_text())
        records = metrics.get("rows", [metrics])
        for row in records:
            family = relative.parts[0]
            if family == "tabular":
                stem = f"{row['env']}_{row['method']}_{row['seed']}"
                checkpoint = path.parent / "checkpoints" / f"{stem}.npz"
                policy = np.load(checkpoint)["policy"]
                env = row["env"]
                name = "tabular_" + stem
            else:
                import torch

                checkpoint = (
                    path.parent / "checkpoints" / ("best.pt" if family == "neural" else "final.pt")
                )
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                policy = payload["policy"].numpy()
                env = "Wireless"
                name = "_".join(relative.parts[:-1])
            if sha256(checkpoint) != row["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint changed: {checkpoint}")
            target = destination / f"{name}.json"
            record = {
                "schema_version": 1,
                "env": env,
                "policy": policy.tolist(),
                "source_metrics": str(path),
                "checkpoint_sha256": row["checkpoint_sha256"],
                "gamma": 0.99,
                "deployment": "Finite-state deterministic action table; all state IDs use the declared environment enumeration.",
            }
            if env == "Wireless":
                record["wireless"] = {
                    "p_data": 0.85,
                    "p_energy": 0.25,
                    "profile": "per_user",
                    "priorities": [1.0, 3.0] if family == "offline" else [1.0, 1.0],
                }
            write_json(target, record)
            _, make, loaded = load_policy(target)
            evidence = rollout_policy(make, loaded, np.arange(3) + 300000, 0.99)
            if family == "tabular":
                source = Path("assets/source_data/tabular") / f"{stem}.npz"
            elif family == "neural":
                source = Path("assets/source_data/neural") / f"{row['recipe']}_{row['seed']}.npz"
            else:
                source = (
                    Path("assets/source_data/offline")
                    / f"epsilon{row['behavior_epsilon']:g}_{row['recipe']}_{row['seed']}.npz"
                )
            reference = np.load(source)
            for key in [
                "returns",
                "discounted_returns",
                "lengths",
                "terminated",
                "truncated",
                "first_trace",
            ]:
                expected = reference[key] if key == "first_trace" else reference[key][:3]
                if not np.array_equal(evidence[key], expected):
                    raise ValueError(f"Fresh policy rollout disagrees: {target}, {key}")
            exported.append(
                {"policy": str(target), "sha256": sha256(target), "fresh_test_episodes": 3}
            )
    write_json("docs/policy_audit.json", {"status": "passed", "exports": exported})
    return exported
