"""Official simulators with explicit, fully observed MiniGrid feature profiles.

RGB frames are for visualization. Policies consume symbolic state, not pixels.
Potential shaping is training-only; final evaluation uses official rewards.
"""

import importlib
import os
from collections import deque

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import numpy as np


def distances(env, target, pass_doors=False):
    """Undiscounted spatial distances; direction and manipulation cost excluded."""
    grid = env.grid
    result = np.full((env.width, env.height), env.width * env.height, dtype=float)
    result[tuple(target)] = 0
    queue = deque([tuple(target)])
    while queue:
        x, y = queue.popleft()
        for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if not (0 <= nx < env.width and 0 <= ny < env.height):
                continue
            obj = grid.get(nx, ny)
            walkable = obj is None or obj.can_overlap() or obj.type == "key"
            walkable |= bool(pass_doors and obj is not None and obj.type == "door")
            if walkable and result[nx, ny] > result[x, y] + 1:
                result[nx, ny] = result[x, y] + 1
                queue.append((nx, ny))
    return result


class NavigationState(gym.Wrapper):
    """Global symbolic map + pose, carrying state and object-relative coordinates.

    All seven original actions remain available. Mission text is constant for these
    single-goal tasks. Remaining time is included because the official terminal
    reward depends on episode duration. No shortest-path features enter observations.
    """

    def __init__(self, env, shaping=False, gamma=0.99, potential_scale=0.05):
        super().__init__(env)
        self.shaping, self.gamma, self.potential_scale = shaping, gamma, potential_scale
        u = self.unwrapped
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(u.width * u.height * 3 + 19,), dtype=np.float32
        )

    def objects(self):
        u = self.unwrapped
        found = {}
        for x in range(u.width):
            for y in range(u.height):
                obj = u.grid.get(x, y)
                if obj is not None and obj.type in ["goal", "key", "door"]:
                    found[obj.type] = (np.array([x, y]), obj)
        return found

    def observation(self):
        u = self.unwrapped
        objects = self.objects()
        pose = np.asarray(u.agent_pos)
        scale = np.array([u.width - 1, u.height - 1])
        carrying = u.carrying is not None and u.carrying.type == "key"
        features = [*(pose / scale), *np.eye(4)[u.agent_dir], float(carrying)]
        for name in ["goal", "key", "door"]:
            obj = objects.get(name)
            features.extend((obj[0] - pose) / scale if obj else [0, 0])
            features.append(float(obj is not None))
        door = objects.get("door")
        features += [
            float(door is not None and door[1].is_open),
            float(door is not None and door[1].is_locked),
            u.step_count / u.max_steps,
        ]
        encoded = u.grid.encode().astype(np.float32)
        encoded /= np.array([10, 5, 2], dtype=np.float32)
        return np.concatenate([encoded.ravel(), features]).astype(np.float32)

    def potential(self):
        u = self.unwrapped
        objects = self.objects()
        pos = tuple(u.agent_pos)
        door = objects.get("door")
        if door is not None and door[1].is_locked:
            # A staged spatial heuristic, not an optimal cost-to-go oracle.
            dpos = tuple(door[0])
            beyond = self.goal_field[dpos]
            if u.carrying is not None and u.carrying.type == "key":
                distance = self.door_field[pos] + 1 + beyond
            else:
                key = objects.get("key")
                if key is None:
                    distance = u.width * u.height
                else:
                    if not np.array_equal(key[0], self.key_position):
                        self.key_field = distances(u, key[0])
                        self.key_position = key[0].copy()
                    distance = self.key_field[pos] + self.door_field[tuple(key[0])] + 2 + beyond
        else:
            distance = self.goal_field[pos]
        return -self.potential_scale * float(distance)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        objects = self.objects()
        u = self.unwrapped
        self.goal_field = distances(u, objects["goal"][0], pass_doors=True)
        if "door" in objects:
            self.door_field = distances(u, objects["door"][0])
        if "key" in objects:
            self.key_field = distances(u, objects["key"][0])
            self.key_position = objects["key"][0].copy()
        self.previous_potential = self.potential()
        return self.observation(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        next_potential = 0.0 if terminated else self.potential()
        shaping = self.gamma * next_potential - self.previous_potential
        self.previous_potential = next_potential
        info = dict(info, official_reward=float(reward), shaping_reward=float(shaping))
        return (
            self.observation(),
            reward + shaping if self.shaping else reward,
            terminated,
            truncated,
            info,
        )


def make_environment(env_id, *, shaping=False, render_mode=None):
    if env_id.startswith("MiniGrid-"):
        importlib.import_module("minigrid")
        return NavigationState(gym.make(env_id, render_mode=render_mode), shaping=shaping)
    if shaping:
        raise ValueError("Shaping is defined only for the declared navigation profile")
    return gym.make(env_id, render_mode=render_mode)
