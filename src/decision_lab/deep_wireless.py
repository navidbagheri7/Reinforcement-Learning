"""Owned masked DQN families on the original energy-harvesting scheduling task."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .artifacts import Run, interval, sha256, write_json
from .mdp import evaluate_policy, value_iteration
from .neural import QNetwork, Replay, greedy_table, quantile_huber, targets
from .tabular import rollout_policy
from .wireless import WirelessEnv, WirelessModel


def fit(config, seed, device="cpu", quick=False, output="results/reference/neural"):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = WirelessModel()
    features = torch.tensor(model.features, dtype=torch.float32, device=device)
    legal = torch.tensor(model.legal, device=device)
    mdp = model.finite_mdp()
    optimal = value_iteration(mdp, config["gamma"])["value"]
    for recipe in config["recipes"]:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        directory = Path(output) / recipe / f"seed{seed}"
        if (directory / "run.json").exists():
            old = json.loads((directory / "run.json").read_text())
            if old["status"] == "complete":
                continue
            raise ValueError("Preserve incomplete output; select a fresh output directory")
        run = Run(
            directory,
            config | {"recipe": recipe, "seed": seed, "device": device, "quick": quick},
            [
                __file__,
                Path(__file__).with_name("neural.py"),
                Path(__file__).with_name("wireless.py"),
            ],
        )
        architecture = {
            "features": features.shape[1],
            "actions": 4,
            "width": 128,
            "dueling": recipe == "double_dueling_per",
            "quantiles": 32 if recipe == "qr_dqn" else 1,
        }
        online = QNetwork(**architecture).to(device)
        target = copy.deepcopy(online).eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(online.parameters(), lr=config["learning_rate"])
        replay = Replay(
            config["replay_capacity"], seed, 0.6 if recipe == "double_dueling_per" else 0.0
        )
        env = WirelessEnv(model)
        _, info = env.reset(seed=seed)
        state = info["state"]
        policy, _ = greedy_table(online, model.features, model.legal, device)
        visits = np.zeros_like(model.legal, dtype=np.int64)
        history, losses, training = [], [], []
        episode_return, episode_length = 0.0, 0
        best = -np.inf
        best_weights = None
        selected_step = 0
        for step in range(1, config["steps"] + 1):
            epsilon = 0.05 + 0.95 * max(0, 1 - step / (0.7 * config["steps"]))
            action = (
                int(rng.choice(np.flatnonzero(model.legal[state])))
                if rng.random() < epsilon
                else int(policy[state])
            )
            _, reward, terminated, truncated, info = env.step(action)
            nxt = info["state"]
            replay.add(state, action, reward, nxt, terminated)
            visits[state, action] += 1
            episode_return += reward
            episode_length += 1
            state = nxt
            if terminated or truncated:
                training.append((step, episode_return, episode_length))
                _, info = env.reset()
                state = info["state"]
                episode_return, episode_length = 0.0, 0
            if step >= config["warmup"] and step % config["train_every"] == 0:
                ids, weights = replay.sample(
                    config["batch_size"], 0.4 + 0.6 * step / config["steps"]
                )
                s = torch.as_tensor(replay.state[ids], device=device)
                ns = torch.as_tensor(replay.next[ids], device=device)
                a = torch.as_tensor(replay.action[ids], device=device)
                r = torch.as_tensor(replay.reward[ids], device=device)
                term = torch.as_tensor(replay.terminated[ids], device=device)
                with torch.no_grad():
                    y = targets(
                        r,
                        term,
                        online(features[ns]),
                        target(features[ns]),
                        legal[ns],
                        config["gamma"],
                        recipe != "dqn",
                    )
                q = online(features[s])[torch.arange(len(ids), device=device), a]
                per_sample = (
                    quantile_huber(q, y)
                    if recipe == "qr_dqn"
                    else F.smooth_l1_loss(q, y, reduction="none")
                )
                loss = (per_sample * torch.as_tensor(weights, device=device)).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
                error = (
                    (q.mean(-1) - y.mean(-1)).detach().cpu().numpy()
                    if recipe == "qr_dqn"
                    else (q - y).detach().cpu().numpy()
                )
                replay.update(ids, error)
                policy, _ = greedy_table(online, model.features, model.legal, device)
                if step % 1000 == 0:
                    losses.append((step, float(loss.detach())))
            if step % config["target_every"] == 0:
                target.load_state_dict(online.state_dict())
            if step % config["validation_every"] == 0 or step == config["steps"]:
                policy, _ = greedy_table(online, model.features, model.legal, device)
                evidence = rollout_policy(
                    lambda: WirelessEnv(model),
                    policy,
                    np.arange(config["validation_episodes"]) + 200000,
                    config["gamma"],
                )
                score = float(evidence["discounted_returns"].mean())
                history.append({"step": step, "discounted_mean": score})
                if score > best:
                    best, selected_step = score, step
                    best_weights = {
                        key: value.detach().cpu().clone()
                        for key, value in online.state_dict().items()
                    }
                print("Neural", recipe, seed, step, score, flush=True)
        env.close()
        online.load_state_dict(best_weights)
        policy, q = greedy_table(online, model.features, model.legal, device)
        checkpoint = directory / "checkpoints" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_weights,
                "architecture": architecture,
                "policy": torch.as_tensor(policy),
                "features": torch.as_tensor(model.features),
                "legal": torch.as_tensor(model.legal),
            },
            checkpoint,
        )
        write_json(
            checkpoint.with_suffix(".json"),
            {
                "sha256": sha256(checkpoint),
                "selected_step": selected_step,
                "seed": seed,
                "recipe": recipe,
                "deployment": "Exported finite-state policy generated on the fitting device; inference uses this table. Float32 critic evaluation on another device may differ at ties.",
            },
        )
        data = rollout_policy(
            lambda: WirelessEnv(model),
            policy,
            np.arange(config["test_episodes"]) + 300000,
            config["gamma"],
        )
        shifted = WirelessModel(p_data=0.95, p_energy=0.15)
        shift = rollout_policy(
            lambda: WirelessEnv(shifted),
            policy,
            np.arange(config["test_episodes"]) + 400000,
            config["gamma"],
        )
        value = evaluate_policy(mdp, policy, config["gamma"])
        destination = directory / "source_data" if quick else Path("assets/source_data/neural")
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / f"{recipe}_{seed}.npz",
            **data,
            q=q,
            policy=policy,
            state_action_visits=visits,
            training_episodes=np.array(training),
            losses=np.array(losses),
            shift_returns=shift["returns"],
            shift_seeds=shift["seeds"],
        )
        run.finish(
            {
                "recipe": recipe,
                "seed": seed,
                "selected_step": selected_step,
                "validation_history": history,
                "test": interval(data["returns"], seed),
                "discounted_test": interval(data["discounted_returns"], seed),
                "shift_test": interval(shift["returns"], seed),
                "normalized_exact_policy_gap": float(
                    (1 - config["gamma"]) * (optimal - value).mean()
                ),
                "packet_loss_fraction": float(data["expired"].sum() / data["offered"].sum()),
                "quantile_crossing_fraction": float((np.diff(q, axis=-1) < 0).mean())
                if recipe == "qr_dqn"
                else None,
                "checkpoint_sha256": sha256(checkpoint),
                "protocol": "Matched interactions; validation-only discounted-return selection; exact model used exclusively for post-fit diagnostics. QR values are return quantiles, not epistemic uncertainty. Continuing time-limit bootstrap retained. Shift uses the fixed fitted policy without adaptation.",
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default="configs/neural.json")
    parser.add_argument("--output", default="results/reference/neural")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.quick:
        config |= {
            "steps": 2048,
            "validation_every": 2048,
            "validation_episodes": 2,
            "test_episodes": 3,
        }
    fit(config, args.seed, args.device, args.quick, args.output)
