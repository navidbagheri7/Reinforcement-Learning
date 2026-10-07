"""Portable experiment records and compact, numeric evidence."""

import hashlib
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, default=serial, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Run:
    def __init__(self, directory, config, sources):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.start = time.perf_counter()
        versions = {}
        for name in ["numpy", "scipy", "gymnasium", "torch", "stable-baselines3"]:
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                pass
        self.record = {
            "status": "running",
            "config": config,
            "python": platform.python_version(),
            "versions": versions,
            "source_sha256": {Path(p).name: sha256(p) for p in sources},
        }
        write_json(self.directory / "run.json", self.record)

    def finish(self, metrics):
        write_json(self.directory / "metrics.json", metrics)
        self.record.update(status="complete", elapsed_seconds=time.perf_counter() - self.start)
        write_json(self.directory / "run.json", self.record)


def interval(values, seed=0):
    x = np.asarray(values, float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite nonempty evidence required")
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(len(x), size=(2000, len(x)))].mean(1)
    return {
        "mean": float(x.mean()),
        "episode_bootstrap_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "n": len(x),
    }
