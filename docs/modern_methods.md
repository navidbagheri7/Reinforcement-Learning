# Distributional control and episodic memory

The extension compares six official-environment presets with three independently
fitted policies each. Every preset uses 1,048,576 interactions, twelve fixed
validation resets and one hundred separate test resets. Maintained SB3 and
SB3-Contrib 2.8.0 implementations are identified separately from the owned wireless
learners. Historical PPO/SAC and full-map navigation remain visible as context.

## Continuous critics

HalfCheetah-v5 TD3 uses two critics, their minimum bootstrapped target, delayed
actor updates, clipped target-policy noise and training exploration noise. TQC
learns two 25-quantile critics, sorts the fifty pooled target atoms and
drops the largest four (the preset is two atoms per critic). Truncation changes the target distribution to reduce
optimistic estimates; return quantiles are not epistemic confidence intervals.
Both use 256-by-256 networks, a million-transition replay buffer, batch 256,
10,000 initial random interactions, one gradient step every four interactions,
Adam learning rate 0.0003, gamma 0.99 and target mixing 0.005. TD3 has actor delay
2, exploration SD 0.1, target noise SD 0.2 and clipping 0.5. TQC uses automatic
entropy temperature. Their critic representations and optimization costs differ.

The old SAC preset shares the interaction/replay/update settings, but trained on
CPU. The extension trained on the declared CUDA device with TF32 disabled and is
deployed on CPU. These are practical reproducible presets, not equal-compute or
paper benchmark reproductions. The original PPO budget/optimization differs.

## Partial observation and recurrent policies

MiniGrid-Empty-8x8-v0 and MiniGrid-DoorKey-6x6-v0 use official sparse rewards and
unchanged dynamics. The policy receives only the official 7-by-7 local symbolic
view (147 scaled object/color/state values) and four direction indicators. It
receives no global map, absolute pose, carrying flag or elapsed step count.
Mission text is constant within each declared task and is omitted. Global RGB
rendering shows the scene to the reader; it is not the policy observation.

PPO and RecurrentPPO use the same 151-value observation, eight environments,
256 steps per environment, batch 256, ten optimization epochs, 128-by-128 policy
and value networks, lambda 0.95 and entropy coefficient 0.01. RecurrentPPO adds
separate 128-unit actor and critic LSTMs. Training interactions match; parameter
counts, recurrence and optimization costs differ. The portable actor resets its
hidden/cell state at every episode start and preserves them within the episode.
The public replay tests verify both behavior and memory against the selected
maintained-library policy. Full-map, shaped-reward and curriculum comparisons
have different information/reward contracts and cannot isolate the memory effect.

## Selection, uncertainty and deployment

Validation uses seeds 200000–200011 after each 65,536 interactions once at least
one optimization update has occurred, plus a check after the final optimizer
update. The first strictly highest mean selects the checkpoint. Only then are
seeds 300000–300099 evaluated. Episode bootstrap intervals condition on a single
trained actor; fitting SD is across three actor means. Neither estimates general
algorithm uncertainty from three fits.
The official Empty-8x8 preset has a fixed layout, start, direction and goal.
Its deterministic actor therefore repeats the same trajectory across seeded
resets, and its conditional episode interval can be exactly zero. This checks a
deterministic navigation task, not generalization over unseen layouts. DoorKey
randomizes its layout, key, door and starting placement using the declared seeds.

Public exports use explicit architectures and finite float32 NPZ weights with
JSON digests. They exclude replay buffers, optimizers and pickle-based training
objects. The actor itself uses NumPy and PyTorch; the official-environment command
uses the `modern` extra. Inference is deterministic, with memory state explicit:

```python
from decision_lab.modern_deployment import load
from decision_lab.modern import environment
actor = load("assets/modern_actors/MiniGrid-DoorKey-6x6-v0_RecurrentPPO_partial_0.npz")
env = environment(actor.metadata["case"])
observation, _ = env.reset(seed=300000)
state = None
start = True
while True:
    action, state = actor.predict(observation, state=state, episode_start=[start])
    observation, reward, terminated, truncated, info = env.step(action)
    start = False
    if terminated or truncated:
        break
env.close()
```

`python scripts/play_modern.py --actor PATH` replays three fixed test episodes.
`validate_modern.py` independently checks selection, all recorded test statistics,
actor/GIF/source identities and separate public deployment commands.
`audit_modern.py` also compares all one hundred exported-actor episodes against
private selected checkpoints and records the first fixed test rollout; it
requires those local checkpoints. The explorer and simulation dashboard require
neither Python nor a GPU. Opening a lesson or explorer does not train a model.

See [measured comparisons](results.md), [public deployment validation](modern_validation.json),
[full selected-model audit](modern_actor_audit.json), and [the explorer](explorer.html).
Primary sources: [TD3](https://arxiv.org/abs/1802.09477),
[TQC](https://arxiv.org/abs/2005.04269),
[maintained TQC documentation](https://sb3-contrib.readthedocs.io/en/v2.8.0/modules/tqc.html),
[recurrent PPO documentation](https://sb3-contrib.readthedocs.io/en/v2.8.0/modules/ppo_recurrent.html).
