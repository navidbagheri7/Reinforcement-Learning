"""Owned finite-action DQN components, distributional loss and prioritized replay."""

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, features, actions, width=128, dueling=False, quantiles=1):
        super().__init__()
        self.actions, self.quantiles, self.dueling = actions, quantiles, dueling
        self.encoder = nn.Sequential(
            nn.Linear(features, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU()
        )
        self.advantage = nn.Linear(width, actions * quantiles)
        if dueling:
            self.value = nn.Linear(width, quantiles)

    def forward(self, x):
        h = self.encoder(x)
        q = self.advantage(h).reshape(-1, self.actions, self.quantiles)
        if self.dueling:
            q = self.value(h)[:, None] + q - q.mean(1, keepdim=True)
        return q.squeeze(-1) if self.quantiles == 1 else q


def targets(rewards, terminated, online_next, target_next, legal, gamma, double=False):
    """Frozen target evaluation; Double DQN selects with online values."""
    with torch.no_grad():
        if online_next.ndim == 3:
            selected = online_next.mean(-1).masked_fill(~legal, -torch.inf).argmax(1)
            next_value = target_next[torch.arange(len(selected), device=selected.device), selected]
            return rewards[:, None] + gamma * (1 - terminated[:, None]) * next_value
        selected = (
            (online_next if double else target_next).masked_fill(~legal, -torch.inf).argmax(1)
        )
        next_value = target_next.gather(1, selected[:, None]).squeeze(1)
        return rewards + gamma * (1 - terminated) * next_value


def quantile_huber(predicted, target, kappa=1.0):
    """Per-transition pairwise quantile Huber, with midpoint quantile levels."""
    n = predicted.shape[-1]
    tau = (torch.arange(n, device=predicted.device) + 0.5) / n
    error = target.detach()[:, None, :] - predicted[:, :, None]
    huber = torch.where(
        error.abs() <= kappa, 0.5 * error.square(), kappa * (error.abs() - 0.5 * kappa)
    )
    return ((tau[None, :, None] - (error.detach() < 0).float()).abs() * huber / kappa).mean((1, 2))


class Replay:
    def __init__(self, capacity, seed=0, alpha=0.0):
        if capacity < 1 or not 0 <= alpha <= 1:
            raise ValueError("Positive replay capacity and alpha in [0,1] required")
        self.capacity, self.alpha = capacity, alpha
        self.rng = np.random.default_rng(seed)
        self.size = 0
        self.cursor = 0
        self.state = np.zeros(capacity, np.int64)
        self.next = np.zeros_like(self.state)
        self.action = np.zeros_like(self.state)
        self.reward = np.zeros(capacity, np.float32)
        self.terminated = np.zeros(capacity, np.float32)
        self.priority = np.ones(capacity)

    def add(self, state, action, reward, nxt, terminated):
        i = self.cursor
        self.state[i] = state
        self.action[i] = action
        self.reward[i] = reward
        self.next[i] = nxt
        self.terminated[i] = terminated
        self.priority[i] = self.priority[: self.size].max() if self.size else 1.0
        self.size = min(self.size + 1, self.capacity)
        self.cursor = (i + 1) % self.capacity

    def sample(self, batch, beta=0.4):
        if not self.size or not 0 <= beta <= 1:
            raise ValueError("Nonempty replay and beta in [0,1] required")
        prob = self.priority[: self.size] ** self.alpha
        prob /= prob.sum()
        ids = self.rng.choice(self.size, size=batch, p=prob)
        weights = (self.size * prob[ids]) ** (-beta)
        weights /= weights.max()
        return ids, weights.astype(np.float32)

    def update(self, ids, error):
        values = np.abs(np.asarray(error, float)) + 1e-6
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite replay priorities")
        merged = np.zeros(self.size)
        np.maximum.at(merged, ids, values)
        touched = np.unique(ids)
        self.priority[touched] = merged[touched]


def greedy_table(model, features, legal, device):
    with torch.no_grad():
        q = model(torch.as_tensor(features, dtype=torch.float32, device=device))
        if not torch.isfinite(q).all():
            raise ValueError("Nonfinite neural action values")
        scores = q.mean(-1) if q.ndim == 3 else q
        policy = scores.masked_fill(~torch.as_tensor(legal, device=device), -torch.inf).argmax(1)
    return policy.cpu().numpy(), q.cpu().numpy()
