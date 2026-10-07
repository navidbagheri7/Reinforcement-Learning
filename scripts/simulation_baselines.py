"""Fixed-seed random and known-model BFS navigation references; no fitting."""

from pathlib import Path

import numpy as np
from PIL import Image

from decision_lab.artifacts import interval, sha256, write_json
from decision_lab.navigation_planner import Planner
from decision_lab.simulation_envs import make_environment


def main():
    rows = []
    root = Path("assets/source_data/simulations")
    for name in ["MiniGrid-FourRooms-v0", "MiniGrid-DoorKey-6x6-v0", "HalfCheetah-v5"]:
        for method in ["random", "BFS"] if name.startswith("MiniGrid") else ["random"]:
            env = make_environment(name, render_mode="rgb_array" if method == "BFS" else None)
            returns, lengths, dones, truncs, frames = [], [], [], [], []
            for i, seed in enumerate(np.arange(100) + 300000):
                obs, _ = env.reset(seed=int(seed))
                env.action_space.seed(int(seed) + 700000)
                planner = Planner(env) if method == "BFS" else None
                total, steps = 0.0, 0
                while True:
                    if method == "BFS" and i == 0:
                        frames.append(Image.fromarray(env.render()))
                    action = planner.predict(obs)[0] if planner else env.action_space.sample()
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total += reward
                    steps += 1
                    if terminated or truncated:
                        if method == "BFS" and i == 0:
                            frames.append(Image.fromarray(env.render()))
                        break
                returns.append(total)
                lengths.append(steps)
                dones.append(terminated)
                truncs.append(truncated)
            env.close()
            path = root / f"{name}_{method}.npz"
            np.savez_compressed(
                path,
                returns=returns,
                lengths=lengths,
                terminated=dones,
                truncated=truncs,
                seeds=np.arange(100) + 300000,
            )
            row = dict(
                env=name,
                method=method,
                test=interval(returns),
                numeric_sha256=sha256(path),
                mean_length=float(np.mean(lengths)),
                success_fraction=float((np.array(returns) > 0).mean())
                if name.startswith("MiniGrid")
                else None,
                protocol="100 fixed test seeds 300000..300099; no training or selection",
                model_access="Known deterministic transition model and full map"
                if method == "BFS"
                else "No model; independently seeded uniform actions",
            )
            rows.append(row)
            if frames:
                gif = Path("assets/rollouts") / f"{name}_BFS.gif"
                frames[0].save(gif, save_all=True, append_images=frames[1:], duration=140, loop=0)
                write_json(
                    gif.with_suffix(".json"),
                    dict(
                        env=name,
                        algorithm="BFS",
                        test_seed=300000,
                        return_value=returns[0],
                        length=lengths[0],
                        terminated=dones[0],
                        truncated=truncs[0],
                        gif_sha256=sha256(gif),
                        source_evidence=str(path),
                        protocol="Fixed first held-out episode; model-based planner, not a learned policy",
                    ),
                )
            print(name, method, row["test"]["mean"], row["success_fraction"], flush=True)
    write_json(
        "docs/simulation_baselines.json",
        dict(
            rows=rows,
            source_sha256={
                "scripts/simulation_baselines.py": sha256(__file__),
                "src/decision_lab/navigation_planner.py": sha256(
                    "src/decision_lab/navigation_planner.py"
                ),
                "src/decision_lab/simulation_envs.py": sha256(
                    "src/decision_lab/simulation_envs.py"
                ),
            },
        ),
    )


if __name__ == "__main__":
    main()
