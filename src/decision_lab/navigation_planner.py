"""Full-map BFS planning reference for the two deterministic MiniGrid tasks.

This baseline receives the known model; it is not a learned RL policy. Orientation,
key pickup, and door unlocking are explicit planning state and action transitions.
"""

from collections import deque

import numpy as np


class Planner:
    def __init__(self, env):
        self.env = env.unwrapped
        self.plan = []

    def search(self):
        u = self.env
        objects = {}
        for x in range(u.width):
            for y in range(u.height):
                obj = u.grid.get(x, y)
                if obj is not None and obj.type in ["key", "door", "goal"]:
                    objects[obj.type] = (x, y)
        door = objects.get("door")
        door_open = bool(door is not None and u.grid.get(*door).is_open)
        carrying = bool(u.carrying is not None and u.carrying.type == "key")
        start = (*u.agent_pos, u.agent_dir, carrying, door_open)
        queue = deque([start])
        parents = {start: None}
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        while queue:
            current = queue.popleft()
            x, y, d, key, opened = current
            if (x, y) == objects["goal"]:
                path = []
                while parents[current] is not None:
                    current, action = parents[current]
                    path.append(action)
                return path[::-1]
            dx, dy = directions[d]
            front = (x + dx, y + dy)
            transitions = [
                (0, (x, y, (d - 1) % 4, key, opened)),
                (1, (x, y, (d + 1) % 4, key, opened)),
            ]
            obj = u.grid.get(*front)
            walkable = obj is None or obj.can_overlap()
            if front == door:
                walkable = opened
            elif front == objects.get("key"):
                walkable = key
            if walkable:
                transitions.append((2, (*front, d, key, opened)))
            if front == objects.get("key") and not key:
                transitions.append((3, (x, y, d, True, opened)))
            if front == door and key and not opened:
                transitions.append((5, (x, y, d, key, True)))
            for action, nxt in transitions:
                if nxt not in parents:
                    parents[nxt] = (current, action)
                    queue.append(nxt)
        raise ValueError("No feasible goal path in declared planning model")

    def predict(self, observation, deterministic=True):
        if self.env.step_count == 0:
            self.plan = self.search()
        if not self.plan:
            raise ValueError("Plan exhausted before a terminal goal")
        return np.int64(self.plan.pop(0)), None
