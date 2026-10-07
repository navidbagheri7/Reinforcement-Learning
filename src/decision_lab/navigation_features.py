"""Task-specific geometry encoding for official 19x19 MiniGrid Four Rooms.

No planner, distances, actions or reward signals enter these features. Four gap
coordinates describe the fixed four-room topology; local walls and relative
objects make spatial decisions easier for a small MLP. Other layouts are rejected.
"""

import torch
from torch.nn import functional as F


def fourrooms_features(observation):
    if observation.ndim != 2 or observation.shape[1] != 19 * 19 * 3 + 19:
        raise ValueError("Four Rooms geometry requires the declared 19x19 observation profile")
    batch = observation.shape[0]
    grid = observation[:, :-19].reshape(batch, 19, 19, 3)
    structured = observation[:, -19:]
    position = torch.round(structured[:, :2] * 18).long()
    goal = torch.round((structured[:, :2] + structured[:, 7:9]) * 18).long()
    objects = torch.round(grid[:, :, :, 0] * 10).long()
    agent_room = (position[:, 0] > 9).long() * 2 + (position[:, 1] > 9).long()
    goal_room = (goal[:, 0] > 9).long() * 2 + (goal[:, 1] > 9).long()
    rooms = torch.cat([F.one_hot(agent_room, 4), F.one_hot(goal_room, 4)], dim=1).float()
    # Each arm contains one traversable gap. Goal tiles can occupy a gap too.
    gap_a = (objects[:, 1:9, 9] != 2).long().argmax(1) + 1
    gap_b = (objects[:, 10:18, 9] != 2).long().argmax(1) + 10
    gap_c = (objects[:, 9, 1:9] != 2).long().argmax(1) + 1
    gap_d = (objects[:, 9, 10:18] != 2).long().argmax(1) + 10
    center = torch.full_like(gap_a, 9)
    gaps = torch.stack([gap_a, center, gap_b, center, center, gap_c, center, gap_d], dim=1)
    relative_gaps = (gaps.reshape(batch, 4, 2) - position[:, None, :]).reshape(batch, 8) / 18
    local = []
    indexes = torch.arange(batch, device=observation.device)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            x = (position[:, 0] + dx).clamp(0, 18)
            y = (position[:, 1] + dy).clamp(0, 18)
            local.append((objects[indexes, x, y] == 2).float())
    return torch.cat([structured, rooms, relative_gaps, torch.stack(local, dim=1)], dim=1)
