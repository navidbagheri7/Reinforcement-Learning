"""Portable, deterministic actor inference without SB3 checkpoint deserialization."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .artifacts import sha256, write_json


class Actor:
    def __init__(self, metadata, arrays):
        self.metadata = metadata
        activation = nn.Tanh if metadata["algorithm"] == "PPO" else nn.ReLU
        dimensions = metadata["dimensions"]
        layers = []
        for i, (a, b) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layer = nn.Linear(a, b, dtype=torch.float32, device="cpu")
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(arrays[f"weight_{i}"]))
                layer.bias.copy_(torch.from_numpy(arrays[f"bias_{i}"]))
            layers.append(layer)
            if i < len(dimensions) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers).eval()

    def predict(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError("The exported actor supports deterministic inference only")
        with torch.no_grad():
            inputs = torch.as_tensor(observation, dtype=torch.float32, device="cpu").unsqueeze(0)
            if self.metadata.get("encoder", "identity") == "fourrooms_geometry_v1":
                from .navigation_features import fourrooms_features

                inputs = fourrooms_features(inputs).to(dtype=torch.float32)
            values = self.net(inputs)[0]
            if self.metadata["action_type"] == "discrete":
                # Match SB3's float32 probability normalization and tie rule.
                probabilities = torch.distributions.Categorical(logits=values).probs
                action = int(probabilities.argmax().item())
            elif self.metadata["algorithm"] == "SAC":
                scaled = values.tanh().numpy()
                low = np.asarray(self.metadata["action_low"], np.float32)
                high = np.asarray(self.metadata["action_high"], np.float32)
                action = low + 0.5 * (scaled + 1.0) * (high - low)
            else:
                action = np.clip(
                    values.numpy(),
                    np.asarray(self.metadata["action_low"], np.float32),
                    np.asarray(self.metadata["action_high"], np.float32),
                )
        return action, None


def export_actor(model, case, target, checkpoint_digest):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    policy = model.policy
    if case["algorithm"] == "PPO":
        hidden = policy.mlp_extractor.policy_net
        output = policy.action_net
    else:
        hidden = policy.actor.latent_pi
        output = policy.actor.mu
    linears = [layer for layer in hidden if isinstance(layer, nn.Linear)] + [output]
    arrays = {}
    for i, layer in enumerate(linears):
        arrays[f"weight_{i}"] = layer.weight.detach().cpu().numpy()
        arrays[f"bias_{i}"] = layer.bias.detach().cpu().numpy()
    np.savez_compressed(target, **arrays)
    discrete = hasattr(model.action_space, "n")
    metadata = dict(
        schema_version=1,
        encoder=getattr(policy.features_extractor, "encoder", "identity"),
        observation_dimension=model.observation_space.shape[0],
        case=case,
        algorithm=case["algorithm"],
        dimensions=[linears[0].in_features] + [layer.out_features for layer in linears],
        action_type="discrete" if discrete else "box",
        activation="tanh" if case["algorithm"] == "PPO" else "relu",
        weights_sha256=sha256(target),
        checkpoint_sha256=checkpoint_digest,
        inference="CPU float32 deterministic actor; no optimizer/critic/pickle",
        observation_factory="decision_lab.simulation_envs.make_environment; shaping disabled",
    )
    if not discrete:
        metadata.update(
            action_low=model.action_space.low.tolist(), action_high=model.action_space.high.tolist()
        )
    write_json(target.with_suffix(".json"), metadata)
    return load_actor(target)


def load_actor(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata.get("schema_version") != 1 or sha256(path) != metadata["weights_sha256"]:
        raise ValueError("Unsupported or changed actor artifact")
    if metadata["algorithm"] not in ["PPO", "SAC"]:
        raise ValueError("Unsupported actor")
    with np.load(path, allow_pickle=False) as arrays:
        if any(not np.isfinite(arrays[key]).all() for key in arrays.files):
            raise ValueError("Nonfinite actor")
        return Actor(metadata, arrays)
