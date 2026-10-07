"""Empirical lecture studies with known values and deliberately exposed failures."""

from pathlib import Path

import numpy as np

from .artifacts import Run, interval


def baird_operators(gamma=0.99):
    phi = np.zeros((7, 8))
    for s in range(6):
        phi[s, s] = 2.0
        phi[s, 7] = 1.0
    phi[6, 6] = 1.0
    phi[6, 7] = 2.0
    phi /= np.sqrt(5.0)
    target = np.zeros((7, 7))
    target[:, 6] = 1.0
    d = np.eye(7) / 7
    covariance = phi.T @ d @ phi
    operator = phi.T @ d @ (np.eye(7) - gamma * target) @ phi
    return phi, covariance, operator


def baird(steps=30000):
    phi, c, a = baird_operators()
    inverse = np.linalg.pinv(c)
    initial = np.ones(8)
    initial[6] = 10.0
    curves = {}
    for name in ["semigradient_td", "gtd2", "lstd"]:
        w = initial.copy()
        h = np.zeros(8)
        trace = []
        for step in range(steps + 1):
            if step % 100 == 0:
                residual = a @ w
                trace.append(
                    (step, np.sqrt(np.mean((phi @ w) ** 2)), residual @ inverse @ residual)
                )
            if name == "semigradient_td":
                w -= 0.05 * (a @ w)
            elif name == "gtd2":
                dw = 0.05 * (a.T @ h)
                dh = 0.1 * (-a @ w - c @ h)
                w += dw
                h += dh
            else:
                w = np.linalg.lstsq(a, np.zeros(8), rcond=None)[0]
        curves[name] = np.array(trace)
    return curves, {
        "minimum_real_operator_eigenvalue": float(np.linalg.eigvals(a).real.min()),
        "covariance_rank": int(np.linalg.matrix_rank(c)),
        "protocol": "Expected-update Baird-style seven-state counterexample. Behavior stationary state distribution uniform; target always transitions to lower state; gamma .99, zero rewards. Known-model expected operators remove sampling noise. LSTD solves the zero-reward reference; no claim of sampled-data superiority. Singular feature covariance uses a pseudoinverse in MSPBE. No clipping of diverging TD.",
    }


def prediction(seeds=30, episodes=300):
    true = np.linspace(-1.0, 1.0, 21)[1:-1]
    arrays = {
        name: np.zeros((seeds, episodes)) for name in ["monte_carlo", "td0", "nstep5", "td_lambda"]
    }
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        values = {name: np.zeros(21) for name in arrays}
        for episode in range(episodes):
            trajectory = [10]
            rewards = []
            while trajectory[-1] not in [0, 20]:
                nxt = trajectory[-1] + rng.choice([-1, 1])
                trajectory.append(nxt)
                rewards.append(-1.0 if nxt == 0 else 1.0 if nxt == 20 else 0.0)
            tmax = len(rewards)
            for name, v in values.items():
                eligibility = np.zeros(21)
                for t, s in enumerate(trajectory[:-1]):
                    if name == "monte_carlo":
                        v[s] += 0.05 * (rewards[-1] - v[s])
                    elif name == "nstep5":
                        end = min(t + 5, tmax)
                        target = sum(rewards[t:end]) + (v[trajectory[end]] if end < tmax else 0.0)
                        v[s] += 0.05 * (target - v[s])
                    else:
                        delta = rewards[t] + v[trajectory[t + 1]] - v[s]
                        eligibility *= 0.8 if name == "td_lambda" else 0.0
                        eligibility[s] += 1.0
                        v += 0.05 * delta * eligibility
                arrays[name][seed, episode] = np.sqrt(np.mean((v[1:-1] - true) ** 2))
    return arrays, {
        "true_values": true.tolist(),
        "fit_seeds": seeds,
        "episodes": episodes,
        "final_rmse": {k: interval(v[:, -1]) for k, v in arrays.items()},
        "protocol": "Identical sampled trajectories shared across methods within each seed; 19 interior states, unbiased random walk, terminal rewards -1/+1, gamma 1, alpha .05, accumulating TD(lambda=.8). Every-visit Monte Carlo; online n-step backups over recorded trajectories. Analytical values used only for error measurement.",
    }


def bandits(runs=100, steps=2000, arms=10, seed=0):
    names = ["epsilon_average", "epsilon_constant", "ucb", "discounted_ucb", "gradient", "thompson"]
    rng = np.random.default_rng(seed)
    initial = rng.normal(size=(runs, arms))
    drift = rng.normal(0, 0.01, size=(runs, steps, arms))
    noise = rng.normal(size=(runs, steps, arms))
    arrays = {}
    metrics = []
    row = np.arange(runs)
    for changing in [False, True]:
        for name in names:
            random = np.random.default_rng(seed + 1)
            estimates = np.zeros((runs, arms))
            counts = np.zeros_like(estimates)
            sums = np.zeros_like(estimates)
            baseline = np.zeros(runs)
            preferences = np.zeros_like(estimates)
            reward_curve = np.zeros((runs, steps))
            regret = np.zeros_like(reward_curve)
            optimal = np.zeros_like(reward_curve)
            truth = initial.copy()
            for t in range(steps):
                if changing:
                    truth += drift[:, t]
                if name == "gradient":
                    probability = np.exp(preferences - preferences.max(1, keepdims=True))
                    probability /= probability.sum(1, keepdims=True)
                    action = (random.random(runs)[:, None] > np.cumsum(probability, axis=1)).sum(1)
                elif name == "thompson":
                    scores = random.normal(sums / (1 + counts), 1 / np.sqrt(1 + counts))
                    action = scores.argmax(1)
                elif name in ["ucb", "discounted_ucb"]:
                    scores = estimates + 2 * np.sqrt(np.log(t + 2) / np.maximum(counts, 1e-12))
                    scores = np.where(counts < 1e-10, np.inf, scores)
                    action = scores.argmax(1)
                else:
                    action = estimates.argmax(1)
                    explore = random.random(runs) < 0.1
                    action[explore] = random.integers(arms, size=explore.sum())
                reward = truth[row, action] + noise[row, t, action]
                reward_curve[:, t] = reward
                regret[:, t] = truth.max(1) - truth[row, action]
                optimal[:, t] = action == truth.argmax(1)
                if name == "discounted_ucb":
                    counts *= 0.99
                    sums *= 0.99
                counts[row, action] += 1
                sums[row, action] += reward
                if name == "gradient":
                    gradient = -probability
                    gradient[row, action] += 1
                    preferences += 0.1 * (reward - baseline)[:, None] * gradient
                    baseline += (reward - baseline) / (t + 1)
                elif name == "epsilon_constant":
                    estimates[row, action] += 0.1 * (reward - estimates[row, action])
                else:
                    estimates = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            stem = ("drifting_" if changing else "stationary_") + name
            arrays[stem + "_rewards"] = reward_curve
            arrays[stem + "_regret"] = regret
            arrays[stem + "_optimal"] = optimal
            metrics.append(
                {
                    "task": "drifting" if changing else "stationary",
                    "method": name,
                    "cumulative_pseudo_regret": interval(regret.sum(1), seed),
                    "last500_reward": interval(reward_curve[:, -500:].mean(1), seed),
                }
            )
    return arrays, {
        "rows": metrics,
        "runs": runs,
        "steps": steps,
        "protocol": "Ten-arm Gaussian reward noise variance1, Gaussian initial means. Drifting means take independent .01-SD random-walk increments. Common arm-specific noise and means across methods. Epsilon .1, constant alpha .1, UCB coefficient2, discounted UCB discount .99, preference alpha .1; Thompson uses conjugate stationary N(0,1) prior with known unit noise. Static posterior is deliberately misspecified on drifting arms. Confidence intervals resample independent bandit instances.",
    }


def run(config, output="results/reference/lectures", quick=False):
    record = Run(output, config | {"quick": quick}, [__file__])
    destination = record.directory / "source_data" if quick else Path("assets/source_data/lectures")
    destination.mkdir(parents=True, exist_ok=True)
    studies = {}
    for name, func in [
        ("baird", lambda: baird(config["baird_steps"])),
        (
            "prediction",
            lambda: prediction(config["prediction_seeds"], config["prediction_episodes"]),
        ),
        ("bandits", lambda: bandits(config["bandit_runs"], config["bandit_steps"])),
    ]:
        data, metrics = func()
        np.savez_compressed(destination / f"{name}.npz", **data)
        studies[name] = metrics
        print("Complete lecture study", name, flush=True)
    record.finish(studies)
