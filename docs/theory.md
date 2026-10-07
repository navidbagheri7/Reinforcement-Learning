# Equations connected to implementations

## Bellman operators and certificates

A finite model stores stochastic transition rows `P[s,a,s_next]`, finite expected rewards, legal masks, and a continuation kernel `C <= P`. True terminal transitions have zero continuation; a time limit does not change C. For discount gamma below one,

`Q_V(s,a) = r(s,a) + gamma sum_next C(s,a,next) V(next)`.

`mdp.py` implements this operator, exact policy evaluation via a linear solve, value iteration and tie-stable policy iteration. A Bellman residual epsilon gives a sup-norm value error bound `epsilon/(1-gamma)`. Finite-horizon backward induction permits gamma one, uses a zero terminal value and exports a policy for each remaining horizon.

The Bellman LP minimizes `mu.T V` subject to `V >= r + gamma C V`. Its dual nonnegative occupancy satisfies `sum_a x(s,a) - gamma sum_(t,a) C(t,a,s)x(t,a) = mu(s)`. A continuing model with stochastic P has total occupancy `1/(1-gamma)`; normalized resource costs multiply by `1-gamma`. Adding budget rows produces randomized constrained policies and shadow prices. Flow residual, primal/dual objective agreement and actual cost are checked, rather than only solver status.

Average-reward stationary occupancy instead satisfies flow balance and unit mass. The gain/bias dual is compared to the occupancy objective. Interpreting its randomized policy globally requires recurrence assumptions; the wireless interior-arrival setting is declared explicitly. Energy conservation yields an equal-priority steady-loss lower bound `2 p_data - 2 p_energy`; weighted priorities and service-floor constraints make separate tradeoffs.

## Sampled control and target separation

Q-learning backs up a legal maximum; SARSA backs up the sampled next behavior action; Expected SARSA integrates the epsilon-greedy distribution. Double Q selects with one table and evaluates with the other. In every case, the target is `reward + gamma (1-terminated) next_value`.

`neural.py` uses the same distinction with frozen target networks. Double DQN chooses the next legal action using online Q and evaluates it using target Q. For online next values [2,1], target [1,9], reward1 and gamma .9, ordinary DQN targets9.1 while Double DQN targets1.9. This numerical oracle is tested. Dueling networks separate value and centered advantages; replay stores actual action IDs. Prioritized replay samples proportional to `(|TD error|+epsilon)^alpha` and applies importance weights.

QR-DQN represents N midpoint quantiles. Its loss averages all predicted/target quantile pairs with asymmetric quantile weights and Huber smoothing. Target quantiles are detached, next action selection uses online quantile means, and target evaluation gathers all quantiles of that action. Quantile ordering is not enforced; measured crossing exposes that limitation.

## Offline learning

Behavior cloning minimizes masked categorical negative log likelihood. Fitted DQN applies sampled Bellman regression to a fixed batch. Discrete CQL adds `alpha * (logsumexp_legal Q(s,a) - Q(s,a_data))`; this lowers relative unsupported-action scores but can be too conservative for useful rare actions.

Discrete IQL fits V using asymmetric squared expectile regression to detached minimum target critics. Critics regress `r + gamma V(next)`; the actor fits logged actions weighted by `exp(beta*(Q_data-V))`, capped at100. Target critics track online critics by EMA. Actor extraction never requires maximization over unobserved actions. Our categorical adaptation and fixed presets are stated explicitly, without inheriting benchmark guarantees from the source papers.

## Linear approximation and instability

MountainCar features use eight offset tilings of normalized position/velocity, disjoint action blocks and replacing eligibility traces. Linear SARSA updates all active trace weights with the TD error and scales its step by the number of tilings. Traces reset on episode reset; truncation still bootstraps the continuing value.

The seven-state Baird-style study defines feature matrix Phi, uniform behavior state distribution D and a target transition to the lower state. `A=Phi.T D (I-gamma P_target) Phi`, `C=Phi.T D Phi`. Expected semigradient TD follows `w <- w-alpha A w`; a negative-real-part eigenvalue exposes instability. Expected GTD2 uses auxiliary h with `w <- w+alpha A.T h` and `h <- h+beta(-A w-C h)`. The singular C is handled with a pseudoinverse in the MSPBE diagnostic. LSTD solves the same zero-reward reference with least squares. Values, not raw nonidentifiable weights, determine the reported error.

## Navigation representations, reward shaping and curricula

Official MiniGrid normally exposes a local symbolic view. The visual navigation extension deliberately uses a full-map state profile with pose, carrying state and elapsed time. Rendering a maze does not mean a policy learns from RGB. Four Rooms has a procedural initial state and doorway layout, so a finite-state action table for one owned maze cannot be transferred directly. The additional geometry encoding computes local walls, room membership and door-relative positions from the same observed map; it does not add a path plan or privileged action. A smaller input dimension also changes network parameter count even when hidden layers and optimizer settings match.

Potential training rewards have the form `F_t = gamma Phi(s_{t+1}) - Phi(s_t)`. Their discounted sum telescopes to `-Phi(s_0) + gamma^T Phi(s_T)`. True terminal states use zero potential. If a time limit is treated as an artificial truncation of a continuing objective, shaped continuation values must be consistent with the retained terminal-observation potential. A time-dependent reward, finite evaluation horizon and imperfect critic make unconditional claims of policy invariance inappropriate. We test reward accounting and preserve the declared bootstrap convention; empirical scores use the untouched official reward.

A curriculum changes the training initial-state distribution, not the held-out scoring task. The Four Rooms curriculum initially samples starts with a small spatial distance to the goal, then widens the range, and finally uses official resets. This creates opportunities to discover successful trajectories before learning long navigation sequences. Validation and test maps are never moved closer to their goals. The wrapper's potential must be recomputed after a training-start change, otherwise the first transition would carry an artificial reward discontinuity.

The known-model BFS planner includes orientation, key pickup and door unlocking in its search state. It is a feasibility/model-access reference, not a sample-trained policy. An RL policy's failure where BFS succeeds can reveal an exploration or representation limit; it does not mean the simulator has no valid route. Every fitted seed and predetermined visual episode remain in the evidence. See [simulation profiles and primary references](simulations.md).

## Twin critics, truncated quantiles and memory

TD3 constructs a smoothed next action by adding clipped Gaussian noise to the target actor. Its detached critic target is `r + gamma * (1-terminated) * min(Q1_target, Q2_target)`. Delayed actor updates optimize the first critic, with soft target updates. This targets overestimation and sensitivity to sharp action-value peaks; it does not supply calibrated uncertainty.

TQC represents each of two target critics by twenty-five quantile atoms. It sorts all fifty next-state atoms and removes the largest four (two per critic preset), leaving forty-six target atoms. Each target also includes the entropy correction `-alpha * log pi(a_next|s_next)`. Quantile Huber regression fits the current critic distributions; the actor uses the mean critic estimate and entropy. Dropping large atoms introduces deliberate pessimism whose strength is a hyperparameter, not a general lower-bound certificate. This continuous maintained-library study complements the owned discrete wireless QR-DQN implementation.

The recurrent MiniGrid policy maintains a hidden/cell state updated by the current local view and direction. In a partially observed task, different histories can produce different actions for the same current observation. Recurrent PPO optimizes clipped policy ratios and bootstrapped advantages while tracking sequence starts and padded sequence masks; those internals belong to SB3-Contrib. Our inference tests check state propagation and episode reset against the fitted actor. Feedforward and recurrent policies share interactions and observed inputs, but memory changes the architecture and training cost. The fixed-map Empty task is a navigation sanity check; randomized DoorKey and all independent fits provide the stronger evaluation. See [the recorded protocol and primary sources](modern_methods.md).
