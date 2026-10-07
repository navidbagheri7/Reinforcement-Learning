# From the original coursework to the rebuilt project

The original MICAS materials cover finite-horizon control, discounted dynamic programming, primal/dual Bellman linear programs, constrained and average-reward control, Monte Carlo/TD control, and linear/deep function approximation. The main project asks for value/policy iteration and Q/deep-Q learning for two users sharing an energy-harvesting wireless link. Original files were archived byte-for-byte; [legacy/manifest.json](../legacy/manifest.json) lists their hashes. They remain local and ignored.

| Source concept or task | Public implementation | Walkthrough |
|---|---|---|
| Finite-horizon, discounted VI/PI and LP | `mdp.py`, `experiments.py` | Exact planning |
| Constrained and average reward | Discounted occupancy constraints, average stationary occupancy | Energy/fairness and recurrence scope |
| TP1 wireless model | `wireless.py`, shared outcome/kernel | Wireless process audit |
| TP2 Q-learning and unfinished deep-Q templates | `tabular.py`, `neural.py`, `deep_wireless.py` | Tabular and neural comparisons |
| MC/TD/n-step and control | `lecture_studies.py`, `tabular.py` | Prediction and official finite tasks |
| Linear function approximation | `approximation.py`, Baird study | MountainCar and off-policy divergence |
| Modern extensions | QR-DQN, discrete CQL/IQL, SB3 PPO/SAC/DQN | Distributional, offline and standard-control studies |

The legacy analytical kernel used uniform increments while the simulator used Bernoulli arrivals. Saturation could lose probability mass; unavailable actions had incompatible infinite versus finite costs; penalties and packet expiry disagreed. A missing deep-learning import and incomplete replay/update templates prevented a complete learning pipeline. The rebuilt kernel and simulator use exactly the same physical outcome function and a finite reward.

The source also mixes a global-channel description with per-user channel state variables. The main model follows the explicit 324-state per-user representation; a separately declared 108-state global-channel interpretation quantifies the ambiguity. Neither is presented as the unique intended interpretation.

For each slot, old unserved packets expire, served packets consume one energy unit, batteries clip to capacity two, independent Bernoulli packets/energy arrive, and channels are independently uniform. Single service needs channel at least one; simultaneous service needs both channels at level two. The global model instead permits user 1, user 2 or both depending on its channel state. Illegal actions receive a finite penalty and idle service; every learning and planning policy uses the same legal mask.

The public repository contains newly authored implementations and English explanations, not translated copies of other people's reports. Original book/slides are private references. New algorithm citations identify their authors in [references.md](references.md).
