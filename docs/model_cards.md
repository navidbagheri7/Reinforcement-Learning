# Model cards and deployment contracts

## Finite-state policies

Tabular and owned neural wireless policies export as schema-version1 deterministic action tables. Each export binds the source metrics and local fitted checkpoint SHA256. The environment ID, wireless physical parameters and complete state enumeration are the contract. `load_policy` checks schema, dimensions, integer actions and every legal mask before deployment.

Neural tables preserve fitting-device selection. Their associated source arrays retain float32 critic values; recomputing a critic on another device can change close argmax ties. A table does not generalize to a different state encoding or action space. Changed arrival probabilities retain the same enumeration and permit fixed-policy sensitivity checks; changed battery capacities do not.

The wireless task is an idealized two-user one-slot packet scheduler, not a production network stack. Physical arrivals are independent, channels iid and capacities small. Exact gaps refer to this mathematical model, not field performance. Offline weighted-priority policies use a different reward objective than equal-priority online policies.

## Owned neural recipes

Two-hidden-layer width128 ReLU critics use normalized physical features. DQN uses target-network legal maximization; Double DQN uses online selection and frozen target evaluation. The combined double/dueling/PER recipe changes three components together and does not isolate their individual causal effects. QR-DQN uses32 midpoint return quantiles and mean-based action selection.

Checkpoints selected by independent repeated validation are compared on100 separate test resets. Their values can be miscalibrated despite a useful policy, particularly when validation selects an early checkpoint under discount .99. The exact post-fit critic-error and Bellman-residual diagnostics are reported separately from deployment-policy gaps. Return quantiles are neither confidence intervals for the true value nor epistemic uncertainty. Raw all-head crossing includes illegal heads; [diagnostics.json](diagnostics.json) separately measures crossing on legal actions.

## Offline recipes

Behavior cloning and discrete IQL export actor argmax; fitted DQN and CQL export critic argmax. All train from the same synthetic logged dataset within each coverage/seed, using fixed final budgets and no environment-based selection. IQL uses independently initialized critics with frozen EMA targets. CQL's fixed coefficient and IQL's expectile/temperature are documented presets, not exhaustive tuned optima.

Behavior data favor user1 even though user2's reward priority is higher. Increasing epsilon changes action-distribution concentration; the measured visited fraction may already include all legal pairs, so this is not automatically an unseen-action benchmark. These experiments do not replicate D4RL or validate theoretical lower bounds for nonlinear finite-training critics. Conservative extraction can retain poor behavior, and naive fitted Q can perform well on this simple structured process.

## Standard control and linear approximation

SB3 2.8.0 owns the PPO/DQN/SAC internals. The local CPU float32 reference uses no observation or reward normalization. `control.json` declares task versions, widths and unequal budgets. Local `best_model.zip` files and `inference.json` bind selected steps and checkpoint hashes. Fresh deployment checks reload the selected files and reproduce held-out episodes and parameter shifts. No full optimizer/environment resume guarantee is made.

Small MountainCar weights are exported under `assets/linear_policies`. They require the exact tiling parameters, action ordering and official observation bounds. Example loading:

```python
import numpy as np
from decision_lab.approximation import Tiles, rollout
payload = np.load('assets/linear_policies/lambda0.9_0.npz')
tiles = Tiles(int(payload['tilings']), int(payload['bins']))
evidence = rollout(payload['weights'], tiles, np.arange(10) + 300000)
print(evidence['returns'].mean())
```

Every parameter shift uses a fixed fitted policy and a preset changed environment. Reported results support a limited sensitivity study, not a general robustness or safety guarantee.

## Official navigation and locomotion actors

Twenty-four independently fitted deterministic actors cover the six presets in `simulations.json` plus spatial-feature and training-curriculum presets in `geometry.json` and `curriculum.json`: fully observed MiniGrid Four Rooms/Door Key PPO with sparse and potential-shaped training, and official HalfCheetah PPO/SAC. All actors are trained from scratch in simulators. Full-map observations and map-derived training shaping are explicitly stronger access than the default local MiniGrid view. Numeric actor weights and preprocessing metadata are public; critics, optimizers and replay buffers remain private.

These actors are evaluated on100 held-out seeds per fit, with complete fresh deployment episodes checked against selected checkpoint evidence. The fixed first-test animations include poor episodes. Three fitted seeds and a small validation set are insufficient for strong general algorithm rankings. The robotics policy acts on simulated state, not camera input or a real robot. See [simulations.md](simulations.md), [gallery](simulation_gallery.md), and [actor audit](simulation_actor_audit.json).
