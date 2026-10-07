# Evaluation protocol and information access

## Selection and held-out resets

Fitting seeds are 0, 1 and 2. Repeated validation resets start at 200000; final test resets at 300000; parameter-shift resets at 400000. Online methods select the best validation checkpoint, using discounted return for continuing wireless and raw episode return for episodic tasks. Control callbacks evaluate only after at least one optimizer update; the final fitted model is evaluated after learning returns, because collection callbacks precede the final update.

Offline methods receive one fixed logged batch per coverage/seed and finish a preset number of gradient updates. They use no environment or transition-model score to select a checkpoint or tune a hyperparameter. All four methods share each batch. Their final policy is evaluated only after fitting. The behavior prefers user 1 although user 2 has priority three; epsilon profiles describe action-distribution concentration rather than guaranteeing unseen state/action pairs. Counted coverage and unsupported deployment occupancy make that distinction measurable.

A fixed first test seed supplies any rollout illustrations. It is not chosen for visual success. Validation is repeatedly observed and can be optimistic; the final independent test is the reported comparison. Training-seed SD and conditional episode bootstrap intervals are separate statistics. Three fitting seeds provide a limited view of variability.

## Information received by each method

Exact planning receives reward and transition tables. Tabular and owned neural control receive state IDs or normalized physical state features, legal masks, rewards and observed next states. They never train against the optimal value function. Exact policy gaps are computed only after fitting, using the full model. Enumerating physical observations for a neural policy cache does not expose transition probabilities or rewards.

Finite-state Gym models come from the installed official `P` dictionaries. Their true termination probability is removed from the continuation operator. Artificial rollout caps keep bootstrap. Taxi's official action mask is used by all compared methods. The owned maze is fully observed and stochastic, not an official Gym benchmark.

Bandit methods share arm-specific noise and true means within each instance. Prediction methods share complete sampled trajectories within each fitting seed; analytical values are evaluation-only. The Baird counterexample intentionally uses expected, known-model linear operators to isolate instability; it is not a sample-efficiency comparison.

## Objectives and budgets

Online wireless rewards count equally weighted expired packets. Offline wireless uses weights [1,3], an explicit different objective. Constrained planning normalizes discounted energy and loss by `(1-gamma)` and uses the declared initial distribution. Average planning assumes the interior-arrival wireless recurrence structure; it is not a general multichain policy theorem.

Tabular and owned neural algorithms have matched interaction budgets within a task. MountainCar compares equal episode budgets and reports the resulting interaction counts. SB3 cases use practical preset budgets and network sizes; PPO/DQN/SAC results are not equal-compute rankings. Framework implementations are credited as such. Runtime includes evaluation and cannot substitute for isolated algorithm throughput.

Standard control shifts use the same fitted checkpoint: LunarLander changes gravity and enables wind; Pendulum changes gravity. Wireless changes packet/energy arrivals. No refitting or checkpoint selection uses shifted results. These are sensitivity checks on a few parameter choices.

## Numerical and deployment limits

Owned wireless fits use float32, frozen target values and legal-action masks. Prioritized replay uses batch-normalized importance weights and merges duplicate priority updates by maximum absolute TD error. QR-DQN selects actions by quantile mean, uses the pairwise quantile Huber loss, and reports crossing. Quantiles describe return randomness; no epistemic calibration is claimed.

Finite-state action tables preserve the fitting-device argmax. Cross-device critic recomputation can differ at close float32 ties, so the deployed table is the reproducible policy contract. CPU SB3 fresh-rollout checks use the same environment/library versions. Seeded execution is not a guarantee of identical results across platforms or library releases. No claim of complete checkpoint/resume replay of environment and optimizer state is made.
