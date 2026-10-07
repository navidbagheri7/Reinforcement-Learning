"""A coherent two-user energy-harvesting scheduler, with two declared channel profiles.

Old unserved packets expire after one slot. New data and energy arrive independently.
A shared outcome function supplies both the simulator and exact stochastic kernel.
"""

from itertools import product

import gymnasium as gym
import numpy as np

from .mdp import FiniteMDP

ACTIONS = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=int)


class WirelessModel:
    def __init__(
        self,
        p_data=0.85,
        p_energy=0.25,
        profile="per_user",
        priorities=(1.0, 1.0),
        invalid_penalty=2.0,
    ):
        if not 0 <= p_data <= 1 or not 0 <= p_energy <= 1 or profile not in ["per_user", "global"]:
            raise ValueError("Valid Bernoulli probabilities and channel profile required")
        self.p_data, self.p_energy, self.profile = p_data, p_energy, profile
        self.priorities = np.asarray(priorities, float)
        self.invalid_penalty = float(invalid_penalty)
        if (
            self.priorities.shape != (2,)
            or not np.isfinite(self.priorities).all()
            or np.any(self.priorities <= 0)
            or not np.isfinite(invalid_penalty)
            or invalid_penalty < 0
        ):
            raise ValueError(
                "Positive priorities and finite nonnegative invalid-action cost required"
            )
        limits = [2, 3, 3, 2, 3, 3] if profile == "per_user" else [2, 3, 2, 3, 3]
        self.states = np.array(list(product(*(range(n) for n in limits))), int)
        self.index = {tuple(row): i for i, row in enumerate(self.states)}
        self.features = self.states / (np.array(limits) - 1)
        self.legal = np.array(
            [[self.is_legal(i, a) for a in range(4)] for i in range(len(self.states))]
        )

    def unpack(self, state):
        row = self.states[state]
        if self.profile == "per_user":
            return row[[0, 3]], row[[1, 4]], row[[2, 5]]
        return row[[0, 2]], row[[1, 3]], row[4:]

    def is_legal(self, state, action):
        d, b, c = self.unpack(state)
        tx = ACTIONS[action]
        if np.any(tx > d) or np.any(tx > b):
            return False
        if self.profile == "per_user":
            return bool(np.all(c[tx > 0] >= (2 if tx.sum() == 2 else 1)))
        allowed = np.array([c[0] in [0, 2], c[0] in [1, 2]])
        return bool(np.all(allowed[tx > 0]))

    def service(self, state, action):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 4:
            raise ValueError("Action ID must be an integer in 0..3")
        d, _, _ = self.unpack(state)
        valid = self.legal[state, action]
        served = ACTIONS[action] if valid else np.zeros(2, dtype=int)
        expired = d - served
        reward = float(-self.priorities @ expired - (0 if valid else self.invalid_penalty))
        return served, expired, reward, valid

    def outcome(self, state, action, data, energy, channel):
        served, expired, reward, valid = self.service(state, action)
        _, battery, _ = self.unpack(state)
        battery = np.minimum(2, battery - served + energy)
        if self.profile == "per_user":
            row = (data[0], battery[0], channel[0], data[1], battery[1], channel[1])
        else:
            row = (data[0], battery[0], data[1], battery[1], channel[0])
        nxt = self.index[row]
        return (
            nxt,
            reward,
            {
                "offered": int(expired.sum() + served.sum()),
                "served": int(served.sum()),
                "expired": int(expired.sum()),
                "served_by_user": served.copy(),
                "invalid_action": not valid,
            },
        )

    def draw(self, rng):
        return (
            rng.binomial(1, self.p_data, 2),
            rng.binomial(1, self.p_energy, 2),
            rng.integers(3, size=2 if self.profile == "per_user" else 1),
        )

    def finite_mdp(self):
        n = len(self.states)
        p = np.zeros((n, 4, n))
        r = np.zeros((n, 4))
        energy = np.zeros((n, 4))
        users = np.zeros((2, n, 4))
        channels = list(product(range(3), repeat=2 if self.profile == "per_user" else 1))
        events = []
        for d in product(range(2), repeat=2):
            pd = np.prod([self.p_data if bit else 1 - self.p_data for bit in d])
            for e in product(range(2), repeat=2):
                pe = np.prod([self.p_energy if bit else 1 - self.p_energy for bit in e])
                for c in channels:
                    events.append((np.array(d), np.array(e), np.array(c), pd * pe / len(channels)))
        for state in range(n):
            for a in range(4):
                served, _, r[state, a], _ = self.service(state, a)
                energy[state, a] = served.sum()
                users[:, state, a] = served
                for d, e, c, prob in events:
                    nxt, _, _ = self.outcome(state, a, d, e, c)
                    p[state, a, nxt] += prob
        self.energy_cost, self.user_service = energy, users
        return FiniteMDP(p, r, self.legal)


class WirelessEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model=None, horizon=512):
        self.model = WirelessModel() if model is None else model
        self.horizon = horizon
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(self.model.features.shape[1],), dtype=np.float32
        )
        self.state_index = 0
        self.elapsed = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state_index = (
            int(self.np_random.integers(len(self.model.states)))
            if not options
            else int(options["state"])
        )
        if not 0 <= self.state_index < len(self.model.states):
            raise ValueError("Unknown initial state")
        self.elapsed = 0
        return self.model.features[self.state_index].astype(np.float32), self.info()

    def info(self):
        return {"state": self.state_index, "action_mask": self.model.legal[self.state_index].copy()}

    def step(self, action):
        nxt, r, info = self.model.outcome(
            self.state_index, action, *self.model.draw(self.np_random)
        )
        self.state_index = nxt
        self.elapsed += 1
        info.update(self.info())
        return (
            self.model.features[nxt].astype(np.float32),
            r,
            False,
            self.elapsed >= self.horizon,
            info,
        )
