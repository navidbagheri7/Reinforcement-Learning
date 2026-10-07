# Primary references

- [Sutton and Barto, Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html): bandits, DP, Monte Carlo, TD, eligibility traces and approximation.
- [Gymnasium environment documentation](https://gymnasium.farama.org/) and [terminated/truncated API rationale](https://farama.org/Gymnasium-Terminated-Truncated-Step-API): official task versions and bootstrap semantics.
- [Stable-Baselines3 2.8.0 documentation](https://stable-baselines3.readthedocs.io/en/v2.8.0/): maintained PPO/DQN/SAC implementations and evaluation guidance.
- [Mnih et al., Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236): DQN.
- [van Hasselt et al., Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461).
- [Wang et al., Dueling Network Architectures](https://arxiv.org/abs/1511.06581).
- [Schaul et al., Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).
- [Dabney et al., Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044).
- [Kumar et al., Conservative Q-Learning](https://arxiv.org/abs/2006.04779).
- [Kostrikov et al., Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169).
- [Sutton et al., Fast Gradient-Descent Methods for Temporal-Difference Learning](https://incompleteideas.net/papers/MSBPSS-09.pdf): gradient TD methods.
- [Schulman et al., Proximal Policy Optimization](https://arxiv.org/abs/1707.06347).
- [Haarnoja et al., Soft Actor-Critic](https://arxiv.org/abs/1801.01290).

The original MICAS slides motivate the learning path and wireless coursework. They remain privately archived. This repository implements declared studies and adaptations; citing a paper does not establish a reproduction of its published benchmark results.

Repository automation follows the current primary [checkout](https://github.com/actions/checkout) and [setup-python](https://github.com/actions/setup-python) instructions. The workflow uses their v7 majors with an explicitly selected Python version and tested PyTorch CPU release.
