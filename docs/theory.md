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
