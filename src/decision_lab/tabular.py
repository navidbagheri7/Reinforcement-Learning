"""Legal-action tabular control and reproducible held-out rollouts."""

import numpy as np


def epsilon_distribution(values, legal, epsilon):
    legal = np.asarray(legal, bool)
    if not legal.any() or not 0 <= epsilon <= 1:
        raise ValueError("Legal actions and epsilon in [0,1] required")
    p = epsilon * legal / legal.sum()
    best = np.flatnonzero(legal)[np.argmax(values[legal])]
    p[best] += 1 - epsilon
    return p


def bootstrap_target(reward, next_value, terminated, gamma):
    """Only true termination suppresses continuation; truncation is not an input."""
    return reward + gamma * (not terminated) * next_value


def state_info(obs, info, actions):
    state = info.get("state", int(obs) if np.asarray(obs).ndim == 0 else None)
    if state is None:
        raise ValueError("Tabular control requires an explicit finite state ID")
    mask = np.asarray(info.get("action_mask", np.ones(actions, bool)), bool)
    return int(state), mask


def rollout_policy(factory, policy, seeds, gamma=0.99):
    env = factory()
    returns = []
    discounted = []
    lengths = []
    termination = []
    truncation = []
    offered = []
    expired = []
    trace = []
    for index, seed in enumerate(seeds):
        obs, info = env.reset(seed=int(seed))
        total = 0.0
        disc = 0.0
        t = 0
        offer = 0
        loss = 0
        while True:
            state, mask = state_info(
                obs, info, len(policy[0]) if np.asarray(policy).ndim == 2 else env.action_space.n
            )
            if np.asarray(policy).ndim == 2:
                action = int(np.argmax(np.where(mask, policy[state], -np.inf)))
            else:
                action = int(policy[state])
            if not mask[action]:
                raise ValueError("Policy selected an illegal action")
            obs, r, term, trunc, info = env.step(action)
            total += r
            disc += gamma**t * r
            t += 1
            offer += info.get("offered", 0)
            loss += info.get("expired", 0)
            if index == 0 and t <= 32:
                trace.append((state, action, r))
            if term or trunc:
                break
        returns.append(total)
        discounted.append(disc)
        lengths.append(t)
        termination.append(term)
        truncation.append(trunc)
        offered.append(offer)
        expired.append(loss)
    env.close()
    return {
        "returns": np.array(returns),
        "discounted_returns": np.array(discounted),
        "lengths": np.array(lengths),
        "terminated": np.array(termination),
        "truncated": np.array(truncation),
        "offered": np.array(offered),
        "expired": np.array(expired),
        "first_trace": np.array(trace),
        "seeds": np.array(seeds),
    }


def learn(
    factory,
    states,
    actions,
    steps,
    algorithm="q_learning",
    seed=0,
    gamma=0.99,
    alpha=0.3,
    validation_every=20000,
    validation_episodes=20,
    wireless=False,
):
    if algorithm not in ["q_learning", "sarsa", "expected_sarsa", "double_q"]:
        raise ValueError("Unknown tabular method")
    rng = np.random.default_rng(seed)
    q = np.zeros((states, actions))
    qb = np.zeros_like(q)
    counts = np.zeros_like(q)
    env = factory()
    obs, info = env.reset(seed=seed)
    state, mask = state_info(obs, info, actions)
    best = -np.inf
    best_q = None
    history = []
    training = []
    episode_return = 0.0
    episode_length = 0

    def exploration(t):
        return 0.05 + 0.95 * max(0.0, 1 - t / (0.8 * steps))

    action = int(
        rng.choice(actions, p=epsilon_distribution(q[state] + qb[state], mask, exploration(0)))
    )
    for step in range(1, steps + 1):
        obs, r, term, trunc, info = env.step(action)
        nxt, next_mask = state_info(obs, info, actions)
        next_action = int(
            rng.choice(
                actions, p=epsilon_distribution(q[nxt] + qb[nxt], next_mask, exploration(step))
            )
        )
        counts[state, action] += 1
        rate = alpha / (1 + counts[state, action] / 1000) ** 0.5
        if algorithm == "double_q":
            left, right = (q, qb) if rng.random() < 0.5 else (qb, q)
            selected = int(np.argmax(np.where(next_mask, left[nxt], -np.inf)))
            target = bootstrap_target(r, right[nxt, selected], term, gamma)
            left[state, action] += rate * (target - left[state, action])
        else:
            if algorithm == "sarsa":
                value = q[nxt, next_action]
            elif algorithm == "expected_sarsa":
                value = epsilon_distribution(q[nxt], next_mask, exploration(step)) @ q[nxt]
            else:
                value = q[nxt, next_mask].max()
            target = bootstrap_target(r, value, term, gamma)
            q[state, action] += rate * (target - q[state, action])
        episode_return += r
        episode_length += 1
        state, mask, action = nxt, next_mask, next_action
        if term or trunc:
            training.append((step, episode_return, episode_length))
            episode_return = 0.0
            episode_length = 0
            obs, info = env.reset()
            state, mask = state_info(obs, info, actions)
            action = int(
                rng.choice(
                    actions, p=epsilon_distribution(q[state] + qb[state], mask, exploration(step))
                )
            )
        if step % validation_every == 0 or step == steps:
            deployed = (q + qb) / 2 if algorithm == "double_q" else q
            evidence = rollout_policy(
                factory, deployed, np.arange(validation_episodes) + 200000, gamma
            )
            scores = evidence["discounted_returns"] if wireless else evidence["returns"]
            score = float(scores.mean())
            history.append(
                {"step": step, "mean_validation_return": score, "returns": scores.tolist()}
            )
            if score > best:
                best = score
                best_q = deployed.copy()
                best_step = step
    env.close()
    return {
        "q": best_q,
        "final_q": (q + qb) / 2 if algorithm == "double_q" else q,
        "counts": counts,
        "validation_history": history,
        "best_validation_return": best,
        "selected_step": best_step,
        "training_episodes": np.array(training).reshape(-1, 3),
    }
