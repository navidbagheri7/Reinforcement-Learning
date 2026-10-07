import numpy as np

from decision_lab.lecture_studies import baird, baird_operators, bandits


def test_off_policy_counterexample_operator():
    phi, c, a = baird_operators()
    assert np.linalg.matrix_rank(c) == 7
    assert np.linalg.eigvals(a).real.min() < -0.01
    curves, _ = baird(10000)
    assert curves["semigradient_td"][-1, 1] > 10 * curves["semigradient_td"][0, 1]
    assert curves["gtd2"][-1, 1] < curves["gtd2"][0, 1]
    assert curves["lstd"][-1, 1] == 0


def test_pseudo_regret_nonnegative():
    data, _ = bandits(3, 20)
    for key, value in data.items():
        assert np.isfinite(value).all()
        if key.endswith("_regret"):
            assert np.all(value >= 0)
