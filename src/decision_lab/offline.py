"""Discrete logged-data comparison: BC, fitted DQN, CQL and IQL.

Fixed final checkpoints and budgets; no deployment evaluation selects a model.
The behavior deliberately favors user 1 despite user 2's higher reward priority.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .artifacts import Run, interval, sha256
from .mdp import evaluate_policy, value_iteration
from .neural import QNetwork, greedy_table, targets
from .tabular import rollout_policy
from .wireless import WirelessEnv, WirelessModel


def independent_critics(architecture, device):
    """Independently initialized critics and genuinely frozen EMA targets."""
    q = QNetwork(**architecture).to(device)
    q2 = QNetwork(**architecture).to(device)
    tq, tq2 = copy.deepcopy(q).eval(), copy.deepcopy(q2).eval()
    for target in [tq, tq2]:
        for parameter in target.parameters():
            parameter.requires_grad_(False)
    return q, q2, tq, tq2


def save_logged(path, batch, coverage):
    """Preserve NPZ bytes and hashes when the logged batch already exists."""
    expected = batch | {"state_action_counts": coverage}
    if path.exists():
        with np.load(path, allow_pickle=False) as previous:
            if set(previous.files) != set(expected) or any(
                not np.array_equal(previous[key], value) for key, value in expected.items()
            ):
                raise ValueError("Different logged dataset exists; choose a fresh output")
    else:
        np.savez_compressed(path, **expected)


def conservative_penalty(q, actions, legal):
    return torch.logsumexp(q.masked_fill(~legal, -torch.inf), 1) - q.gather(
        1, actions[:, None]
    ).squeeze(1)


def expectile_loss(value, target, tau):
    error = target.detach() - value
    return torch.where(error > 0, tau, 1 - tau) * error.square()


def logged_batch(model, count, epsilon, seed):
    env = WirelessEnv(model)
    rng = np.random.default_rng(seed)
    _, info = env.reset(seed=seed)
    state = info["state"]
    rows = []
    for _ in range(count):
        preferred = 1 if model.legal[state, 1] else 0
        action = (
            int(rng.choice(np.flatnonzero(model.legal[state])))
            if rng.random() < epsilon
            else preferred
        )
        _, reward, term, trunc, info = env.step(action)
        rows.append((state, action, reward, info["state"], int(term)))
        state = info["state"]
        if term or trunc:
            _, info = env.reset()
            state = info["state"]
    env.close()
    rows = np.asarray(rows)
    return {
        "state": rows[:, 0].astype(np.int64),
        "action": rows[:, 1].astype(np.int64),
        "reward": rows[:, 2].astype(np.float32),
        "next": rows[:, 3].astype(np.int64),
        "terminated": rows[:, 4].astype(np.float32),
    }


def fit(config, seed, device="cpu", quick=False, output="results/reference/offline"):
    from .protocols import completed_batch

    if completed_batch("offline", config, seed, device, quick, output):
        print("Already complete; preserved logged evidence")
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = WirelessModel(priorities=(1.0, 3.0))
    features = torch.tensor(model.features, dtype=torch.float32, device=device)
    legal = torch.tensor(model.legal, device=device)
    mdp = model.finite_mdp()
    optimum = value_iteration(mdp, config["gamma"])["value"]
    for epsilon in config["behavior_epsilon"]:
        label = f"epsilon{epsilon:g}"
        batch = logged_batch(model, config["dataset_steps"], epsilon, 10000 + seed)
        coverage = np.zeros_like(model.legal, dtype=np.int64)
        np.add.at(coverage, (batch["state"], batch["action"]), 1)
        destination = (
            Path(output) / label / f"seed{seed}" / "source_data"
            if quick
            else Path("assets/source_data/offline")
        )
        destination.mkdir(parents=True, exist_ok=True)
        save_logged(destination / f"logged_{label}_{seed}.npz", batch, coverage)
        tensors = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
        for recipe in config["recipes"]:
            directory = Path(output) / label / recipe / f"seed{seed}"
            if (directory / "run.json").exists():
                old = json.loads((directory / "run.json").read_text())
                if old["status"] == "complete":
                    continue
                raise ValueError("Preserve incomplete output; choose a fresh directory")
            run = Run(
                directory,
                config
                | {
                    "seed": seed,
                    "behavior_epsilon": epsilon,
                    "recipe": recipe,
                    "quick": quick,
                    "device": device,
                },
                [
                    __file__,
                    Path(__file__).with_name("neural.py"),
                    Path(__file__).with_name("wireless.py"),
                ],
            )
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            architecture = {"features": features.shape[1], "actions": 4, "width": 128}
            q, q2, tq, tq2 = independent_critics(architecture, device)
            actor = QNetwork(**architecture).to(device)
            value = QNetwork(features.shape[1], 1, 128).to(device)
            opt = torch.optim.Adam(q.parameters(), lr=config["learning_rate"])
            opt2 = torch.optim.Adam(q2.parameters(), lr=config["learning_rate"])
            optv = torch.optim.Adam(value.parameters(), lr=config["learning_rate"])
            opta = torch.optim.Adam(actor.parameters(), lr=config["learning_rate"])
            history = []
            for update in range(1, config["updates"] + 1):
                ids = rng.integers(config["dataset_steps"], size=config["batch_size"])
                s, a, r, ns, term = (
                    tensors[k][ids] for k in ["state", "action", "reward", "next", "terminated"]
                )
                x, xn = features[s], features[ns]
                if recipe == "bc":
                    loss = F.cross_entropy(q(x).masked_fill(~legal[s], -torch.inf), a)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                elif recipe == "iql":
                    with torch.no_grad():
                        observed = torch.minimum(tq(x), tq2(x)).gather(1, a[:, None]).squeeze(1)
                    v = value(x).squeeze(1)
                    lv = expectile_loss(v, observed, config["expectile"]).mean()
                    optv.zero_grad()
                    lv.backward()
                    optv.step()
                    with torch.no_grad():
                        y = r + config["gamma"] * (1 - term) * value(xn).squeeze(1)
                    lq = F.mse_loss(q(x).gather(1, a[:, None]).squeeze(1), y)
                    lq2 = F.mse_loss(q2(x).gather(1, a[:, None]).squeeze(1), y)
                    opt.zero_grad()
                    lq.backward()
                    opt.step()
                    opt2.zero_grad()
                    lq2.backward()
                    opt2.step()
                    with torch.no_grad():
                        weight = torch.exp(
                            config["advantage_temperature"] * (observed - value(x).squeeze(1))
                        ).clamp(max=100.0)
                    la = (
                        weight
                        * F.cross_entropy(
                            actor(x).masked_fill(~legal[s], -torch.inf), a, reduction="none"
                        )
                    ).mean()
                    opta.zero_grad()
                    la.backward()
                    opta.step()
                    with torch.no_grad():
                        for target, online in [(tq, q), (tq2, q2)]:
                            for tp, op in zip(target.parameters(), online.parameters()):
                                tp.lerp_(op, 0.005)
                    loss = lq + lv + la
                else:
                    with torch.no_grad():
                        y = targets(r, term, q(xn), tq(xn), legal[ns], config["gamma"], True)
                    scores = q(x)
                    loss = F.smooth_l1_loss(scores.gather(1, a[:, None]).squeeze(1), y)
                    if recipe == "cql":
                        loss = (
                            loss
                            + config["conservative_alpha"]
                            * conservative_penalty(scores, a, legal[s]).mean()
                        )
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                    opt.step()
                    if update % config["target_every"] == 0:
                        tq.load_state_dict(q.state_dict())
                if update % 1000 == 0 or update == config["updates"]:
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite offline objective")
                    history.append((update, float(loss.detach())))
            deployment = actor if recipe == "iql" else q
            policy, scores = greedy_table(deployment, model.features, model.legal, device)
            checkpoint = directory / "checkpoints" / "final.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in deployment.state_dict().items()},
                    "architecture": architecture,
                    "policy": torch.as_tensor(policy),
                    "critic": q.state_dict(),
                    "value": value.state_dict() if recipe == "iql" else None,
                },
                checkpoint,
            )
            evidence = rollout_policy(
                lambda: WirelessEnv(model),
                policy,
                np.arange(config["test_episodes"]) + 300000,
                config["gamma"],
            )
            shifted = WirelessModel(0.95, 0.15, priorities=(1.0, 3.0))
            shift = rollout_policy(
                lambda: WirelessEnv(shifted),
                policy,
                np.arange(config["test_episodes"]) + 400000,
                config["gamma"],
            )
            exact = evaluate_policy(mdp, policy, config["gamma"])
            p = mdp.transition[np.arange(mdp.states), policy]
            occupancy = (1 - config["gamma"]) * np.linalg.solve(
                np.eye(mdp.states) - config["gamma"] * p.T, np.ones(mdp.states) / mdp.states
            )
            np.savez_compressed(
                destination / f"{label}_{recipe}_{seed}.npz",
                **evidence,
                policy=policy,
                scores=scores,
                losses=np.array(history),
                shift_returns=shift["returns"],
                shift_seeds=shift["seeds"],
            )
            run.finish(
                {
                    "recipe": recipe,
                    "seed": seed,
                    "behavior_epsilon": epsilon,
                    "test": interval(evidence["returns"], seed),
                    "discounted_test": interval(evidence["discounted_returns"], seed),
                    "shift_test": interval(shift["returns"], seed),
                    "normalized_exact_policy_gap": float(
                        (1 - config["gamma"]) * (optimum - exact).mean()
                    ),
                    "visited_legal_fraction": float((coverage[model.legal] > 0).mean()),
                    "unsupported_deployment_occupancy": float(
                        occupancy[coverage[np.arange(mdp.states), policy] == 0].sum()
                    ),
                    "checkpoint_sha256": sha256(checkpoint),
                    "dataset_sha256": sha256(destination / f"logged_{label}_{seed}.npz"),
                    "protocol": "Pure offline fixed final budget. Identical logged batch per coverage/seed shared across methods. No environment/model checkpoint or hyperparameter selection. BC/IQL deploy masked argmax logits; CQL/fitted DQN deploy masked argmax critic. Weighted priorities [1,3] are a declared extension.",
                }
            )
            print(
                "Offline",
                label,
                recipe,
                seed,
                float((1 - config["gamma"]) * (optimum - exact).mean()),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default="configs/offline.json")
    parser.add_argument("--output", default="results/reference/offline")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.quick:
        config |= {
            "dataset_steps": 1000,
            "updates": 20,
            "test_episodes": 3,
            "behavior_epsilon": [0.05],
        }
    fit(config, args.seed, args.device, args.quick, args.output)
