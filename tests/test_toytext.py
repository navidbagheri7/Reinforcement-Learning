import numpy as np

from decision_lab.tabular import bootstrap_target
from decision_lab.toytext import factory, model_from_env


def test_cliff_time_limit_continues_bootstrap():
    env = factory("CliffWalking-v1")()
    env.reset(seed=4)
    for _ in range(500):
        obs, reward, terminated, truncated, info = env.step(3)
    assert not terminated and truncated
    assert bootstrap_target(1.0, 10.0, False, 0.9) == 10.0
    env.close()


def test_frozen_lake_terminal_has_no_continuation():
    env = factory("FrozenLake-v1")()
    mdp = model_from_env(env)
    assert np.all(mdp.continuation[15] == 0)
    assert np.allclose(mdp.transition.sum(2), 1)
    env.close()
