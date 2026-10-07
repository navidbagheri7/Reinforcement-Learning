"""Official Gymnasium finite models and an explicitly owned fully observed maze."""

import gymnasium as gym
import numpy as np

from .mdp import FiniteMDP


class MazeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, horizon=256, slip=0.1):
        self.layout = [
            "S....#.....",
            "###..#.###.",
            ".....#...#.",
            ".###.###.#.",
            "...#.....#.",
            ".#.#####.#.",
            ".#.......#.",
            ".#######.#.",
            ".........G.",
        ]
        self.positions = [
            (i, j) for i, row in enumerate(self.layout) for j, c in enumerate(row) if c != "#"
        ]
        self.index = {p: i for i, p in enumerate(self.positions)}
        self.goal = self.index[(8, 9)]
        self.start = 0
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(len(self.positions))
        self.horizon, self.slip = horizon, slip
        self.P = {}
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for s, (i, j) in enumerate(self.positions):
            self.P[s] = {}
            for a in range(4):
                if s == self.goal:
                    self.P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                entries = []
                for used, p in [(a, 1 - slip), ((a - 1) % 4, slip / 2), ((a + 1) % 4, slip / 2)]:
                    di, dj = directions[used]
                    nxt = self.index.get((i + di, j + dj), s)
                    done = nxt == self.goal
                    entries.append((p, nxt, 1.0 if done else -0.01, done))
                self.P[s][a] = entries

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start
        self.elapsed = 0
        return self.state, {}

    def step(self, action):
        choices = self.P[self.state][int(action)]
        p, nxt, r, term = choices[
            self.np_random.choice(len(choices), p=[row[0] for row in choices])
        ]
        self.state = nxt
        self.elapsed += 1
        return nxt, r, term, self.elapsed >= self.horizon, {}


def factory(name):
    if name == "Maze":
        return MazeEnv
    kwargs = {"map_name": "4x4", "is_slippery": True} if name == "FrozenLake-v1" else {}
    if name == "CliffWalking-v1":
        kwargs["max_episode_steps"] = 500
    return lambda: gym.make(name, **kwargs)


def model_from_env(env):
    core = env.unwrapped
    n = env.observation_space.n
    a = env.action_space.n
    p = np.zeros((n, a, n))
    cont = np.zeros_like(p)
    r = np.zeros((n, a))
    legal = np.ones((n, a), bool)
    for s in range(n):
        for action in range(a):
            for prob, nxt, reward, done in core.P[s][action]:
                p[s, action, nxt] += prob
                r[s, action] += prob * reward
                if not done:
                    cont[s, action, nxt] += prob
        if hasattr(core, "action_mask"):
            legal[s] = core.action_mask(s).astype(bool)
    return FiniteMDP(p, r, legal, cont)
