"""Minimal deterministic actors with portable NPZ weights and explicit memory.

No SB3 classes or replay buffers are needed for inference. Supports the four
declared extension policies, rejects unknown schemas and checks every weight.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .artifacts import sha256, write_json


def multilayer(input_size, widths, output_size, activation, squash=False):
    layers = []
    for width in widths:
        layers.extend(
            [nn.Linear(input_size, width, dtype=torch.float32, device="cpu"), activation()]
        )
        input_size = width
    layers.append(nn.Linear(input_size, output_size, dtype=torch.float32, device="cpu"))
    if squash:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Actor:
    def __init__(self, metadata, weights):
        if metadata.get("schema_version") != 1 or metadata["algorithm"] not in {
            "PPO",
            "RecurrentPPO",
            "TD3",
            "TQC",
        }:
            raise ValueError("Unsupported extension actor")
        self.metadata = metadata
        self.algorithm = metadata["algorithm"]
        self.inputs = metadata["inputs"]
        self.outputs = metadata["outputs"]
        self.network = multilayer(
            128 if self.algorithm == "RecurrentPPO" else self.inputs,
            [128, 128] if self.algorithm in {"PPO", "RecurrentPPO"} else [256, 256],
            self.outputs,
            nn.Tanh if self.algorithm in {"PPO", "RecurrentPPO"} else nn.ReLU,
            squash=self.algorithm == "TD3",
        ).eval()
        modules = {"network": self.network}
        if self.algorithm == "RecurrentPPO":
            self.lstm = nn.LSTM(
                self.inputs, 128, num_layers=1, dtype=torch.float32, device="cpu"
            ).eval()
            modules["lstm"] = self.lstm
        expected = {
            f"{prefix}.{key}" for prefix, module in modules.items() for key in module.state_dict()
        }
        if set(weights) != expected:
            raise ValueError("Weight keys disagree with declared architecture")
        for prefix, module in modules.items():
            payload = {}
            for key, tensor in module.state_dict().items():
                array = weights[f"{prefix}.{key}"]
                if (
                    array.dtype != np.float32
                    or array.shape != tuple(tensor.shape)
                    or not np.isfinite(array).all()
                ):
                    raise ValueError("Invalid actor weight shape/type/value")
                payload[key] = torch.from_numpy(array.copy())
            module.load_state_dict(payload)

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        if not deterministic:
            raise ValueError("Portable actors implement deterministic deployment only")
        obs = np.asarray(observation, np.float32)
        if obs.shape != (self.inputs,) or not np.isfinite(obs).all():
            raise ValueError("A finite single environment observation is required")
        memory = None
        with torch.no_grad():
            features = torch.from_numpy(obs).reshape(1, -1)
            if self.algorithm == "RecurrentPPO":
                if state is None or (
                    episode_start is not None and bool(np.asarray(episode_start).reshape(-1)[0])
                ):
                    state = (np.zeros((1, 1, 128), np.float32), np.zeros((1, 1, 128), np.float32))
                if len(state) != 2 or any(
                    np.asarray(s).shape != (1, 1, 128) or not np.isfinite(s).all() for s in state
                ):
                    raise ValueError("Invalid recurrent memory")
                features, next_state = self.lstm(
                    features.unsqueeze(0),
                    tuple(torch.as_tensor(s, dtype=torch.float32, device="cpu") for s in state),
                )
                features = features.squeeze(0)
                memory = tuple(s.numpy() for s in next_state)
            output = self.network(features)
            if self.algorithm == "TQC":
                output = torch.tanh(output)
            if self.algorithm in {"PPO", "RecurrentPPO"}:
                # Match SB3's float32 probability normalization and tie rule.
                probabilities = torch.distributions.Categorical(logits=output).probs
                return np.asarray(probabilities.argmax(dim=1).item()), memory
            output = output.numpy()[0]
            low = np.asarray(self.metadata["low"], dtype=np.float32)
            high = np.asarray(self.metadata["high"], dtype=np.float32)
            action = low + 0.5 * (output + 1) * (high - low)
            return action.astype(np.float32), None


def export(model, case, destination, checkpoint_sha256):
    policy = model.policy
    algorithm = case["algorithm"]
    metadata = dict(
        schema_version=1,
        algorithm=algorithm,
        inputs=int(model.observation_space.shape[0]),
        outputs=int(model.action_space.n)
        if algorithm in {"PPO", "RecurrentPPO"}
        else int(model.action_space.shape[0]),
        case=case,
        checkpoint_sha256=checkpoint_sha256,
        memory="Reset on episode_start; persist within an episode"
        if algorithm == "RecurrentPPO"
        else "stateless",
    )
    if algorithm in {"PPO", "RecurrentPPO"}:
        network = nn.Sequential(
            *list(policy.mlp_extractor.policy_net.children()), policy.action_net
        )
    elif algorithm == "TD3":
        network = policy.actor.mu
    else:
        network = nn.Sequential(*list(policy.actor.latent_pi.children()), policy.actor.mu)
    weights = {
        "network." + key: value.detach().cpu().numpy()
        for key, value in network.state_dict().items()
    }
    if algorithm == "RecurrentPPO":
        weights.update(
            {
                "lstm." + key: value.detach().cpu().numpy()
                for key, value in policy.lstm_actor.state_dict().items()
            }
        )
    if algorithm in {"TD3", "TQC"}:
        metadata.update(low=model.action_space.low.tolist(), high=model.action_space.high.tolist())
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **weights)
    metadata["weights_sha256"] = sha256(destination)
    write_json(destination.with_suffix(".json"), metadata)
    return load(destination)


def load(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != metadata["weights_sha256"]:
        raise ValueError("Actor weight digest mismatch")
    with np.load(path, allow_pickle=False) as payload:
        return Actor(metadata, dict(payload))
