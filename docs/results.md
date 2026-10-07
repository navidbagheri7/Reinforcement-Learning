# Measured comparisons

74 completed reference protocols. Every fit is listed in [individual_fits.csv](individual_fits.csv); [comparison.csv](comparison.csv) aggregates the fitting seeds.

Bars show the mean of fitting-seed test means; dots show every fit. Fit SD measures training variability across only three fits. Episode bootstrap intervals in the individual table quantify resets conditional on one fitted policy; they do not measure training uncertainty. No test return selects a checkpoint or hyperparameter.

| Task | Method | Fits | Mean test return | Fit SD | Worst fit | Mean exact gap |
|---|---|---:|---:|---:|---:|---:|
| MountainCar-v0 | SARSA(lambda=0) | 3 | -100.31 | 0.95 | -100.90 | — |
| MountainCar-v0 | SARSA(lambda=0.9) | 3 | -102.14 | 0.56 | -102.78 | — |
| CartPole-v1 | DQN | 3 | 290.77 | 181.28 | 180.50 | — |
| CartPole-v1 | PPO | 3 | 500.00 | 0.00 | 500.00 | — |
| LunarLander-v3 | DQN | 3 | 262.43 | 6.74 | 258.06 | — |
| LunarLander-v3 | PPO | 3 | 220.53 | 11.69 | 207.45 | — |
| Pendulum-v1 | PPO | 3 | -184.97 | 13.93 | -200.98 | — |
| Pendulum-v1 | SAC | 3 | -152.20 | 0.43 | -152.49 | — |
| Wireless online | double_dqn | 3 | -625.68 | 0.65 | -626.43 | 0.00089 |
| Wireless online | double_dueling_per | 3 | -625.44 | 0.20 | -625.67 | 0.00026 |
| Wireless online | dqn | 3 | -625.31 | 0.02 | -625.34 | 0.00000 |
| Wireless online | qr_dqn | 3 | -625.38 | 0.13 | -625.53 | 0.00012 |
| Wireless offline epsilon=0.05 | bc | 3 | -1613.43 | 0.00 | -1613.43 | 0.72354 |
| Wireless offline epsilon=0.05 | cql | 3 | -1613.43 | 0.00 | -1613.43 | 0.72354 |
| Wireless offline epsilon=0.05 | fitted_dqn | 3 | -1246.62 | 0.00 | -1246.62 | 0.00113 |
| Wireless offline epsilon=0.05 | iql | 3 | -1248.29 | 1.42 | -1249.88 | 0.00495 |
| Wireless offline epsilon=0.4 | bc | 3 | -1613.43 | 0.00 | -1613.43 | 0.72354 |
| Wireless offline epsilon=0.4 | cql | 3 | -1597.23 | 28.06 | -1613.43 | 0.69333 |
| Wireless offline epsilon=0.4 | fitted_dqn | 3 | -1285.30 | 22.02 | -1301.47 | 0.09082 |
| Wireless offline epsilon=0.4 | iql | 3 | -1246.48 | 0.25 | -1246.62 | 0.00100 |
| Wireless offline epsilon=1 | bc | 3 | -1448.61 | 143.84 | -1614.11 | 0.40677 |
| Wireless offline epsilon=1 | cql | 3 | -1293.60 | 9.91 | -1304.91 | 0.10651 |
| Wireless offline epsilon=1 | fitted_dqn | 3 | -1280.79 | 19.84 | -1297.29 | 0.07922 |
| Wireless offline epsilon=1 | iql | 3 | -1246.26 | 0.62 | -1246.62 | 0.00075 |
| Wireless | q_learning | 3 | -646.43 | 2.04 | -648.69 | 0.04559 |
| Wireless | sarsa | 3 | -642.27 | 4.32 | -645.32 | 0.03599 |
| Wireless | expected_sarsa | 3 | -647.65 | 1.30 | -648.83 | 0.04744 |
| Wireless | double_q | 3 | -648.12 | 3.04 | -651.09 | 0.04951 |
| FrozenLake-v1 | q_learning | 3 | 0.73 | 0.00 | 0.73 | -0.00000 |
| FrozenLake-v1 | sarsa | 3 | 0.63 | 0.10 | 0.52 | 0.00095 |
| FrozenLake-v1 | expected_sarsa | 3 | 0.71 | 0.04 | 0.66 | 0.00026 |
| FrozenLake-v1 | double_q | 3 | 0.63 | 0.11 | 0.51 | 0.00081 |
| CliffWalking-v1 | q_learning | 3 | -13.00 | 0.00 | -13.00 | -0.00000 |
| CliffWalking-v1 | sarsa | 3 | -17.00 | 0.00 | -17.00 | 0.03458 |
| CliffWalking-v1 | expected_sarsa | 3 | -17.00 | 0.00 | -17.00 | 0.03458 |
| CliffWalking-v1 | double_q | 3 | -13.00 | 0.00 | -13.00 | -0.00000 |
| Taxi-v3 | q_learning | 3 | 6.66 | 2.34 | 3.95 | 0.00342 |
| Taxi-v3 | sarsa | 3 | 2.56 | 1.19 | 1.86 | 0.02186 |
| Taxi-v3 | expected_sarsa | 3 | 7.34 | 1.18 | 5.97 | 0.00685 |
| Taxi-v3 | double_q | 3 | 8.02 | 0.00 | 8.02 | 0.00002 |
| Maze | q_learning | 3 | 0.82 | 0.00 | 0.82 | 0.00001 |
| Maze | sarsa | 3 | 0.82 | 0.00 | 0.82 | 0.00001 |
| Maze | expected_sarsa | 3 | 0.82 | 0.00 | 0.82 | 0.00000 |
| Maze | double_q | 3 | 0.82 | 0.00 | 0.82 | -0.00000 |

Exact normalized gaps use the full discounted MDP from the declared initial distribution. They measure a continuing/model-defined objective, while time-limited test episode returns measure finite rollouts. Wireless offline priorities [1,3] differ from online [1,1], so their raw returns must not be compared across those tasks.

Training budgets differ across PPO/DQN/SAC control cases and are recorded in configs and run records. These compare practical preset protocols, not equal-compute algorithm rankings. Planning methods receive the exact transition model; learned methods receive sampled transitions.

CQL/IQL are discrete adaptations on a small known-action-mask wireless environment, not reproductions of the papers’ D4RL benchmark claims. Limited behavior coverage can make conservative policy extraction retain the biased behavior. Quantile crossing is reported for QR-DQN; return quantiles do not certify epistemic uncertainty.

Plots rebuild from numeric NPZ arrays and JSON histories using `decision-lab report`; every plot has PNG and PDF versions. Scientific figures use fixed test protocols, including poor fitted seeds.
