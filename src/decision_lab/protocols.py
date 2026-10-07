"""Prevent public commands from mixing protocol versions or overwriting logged data."""

import json
from pathlib import Path


def completed_batch(command, config, seed, device, quick, output):
    root = Path(output)
    if command == "neural":
        expected = [
            (
                root / recipe / f"seed{seed}",
                config | {"recipe": recipe, "seed": seed, "device": device, "quick": quick},
            )
            for recipe in config["recipes"]
        ]
    elif command == "offline":
        expected = [
            (
                root / f"epsilon{epsilon:g}" / recipe / f"seed{seed}",
                config
                | {
                    "seed": seed,
                    "behavior_epsilon": epsilon,
                    "recipe": recipe,
                    "device": device,
                    "quick": quick,
                },
            )
            for epsilon in config["behavior_epsilon"]
            for recipe in config["recipes"]
        ]
    else:
        return False
    present = []
    for directory, identity in expected:
        path = directory / "run.json"
        if not path.exists():
            present.append(False)
            continue
        record = json.loads(path.read_text())
        if record["status"] != "complete" or record["config"] != identity:
            raise ValueError(
                f"Preserve incomplete/different protocol at {directory}; use a new --output directory"
            )
        present.append(True)
    if any(present) and not all(present):
        raise ValueError(
            "Preserve this partially completed batch; use a new --output directory rather than rewriting shared logged evidence"
        )
    return all(present)
