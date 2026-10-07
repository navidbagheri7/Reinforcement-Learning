import numpy as np
import torch

from decision_lab.neural import Replay, quantile_huber, targets
from decision_lab.tabular import bootstrap_target


def test_double_target_oracle_mask_and_true_termination():
    online = torch.tensor([[2.0, 1.0, 99.0], [2.0, 1.0, 99.0]])
    target = torch.tensor([[1.0, 9.0, 1000.0], [1.0, 9.0, 1000.0]], requires_grad=True)
    legal = torch.tensor([[True, True, False]] * 2)
    standard = targets(torch.ones(2), torch.tensor([0.0, 1.0]), online, target, legal, 0.9)
    double = targets(torch.ones(2), torch.tensor([0.0, 1.0]), online, target, legal, 0.9, True)
    torch.testing.assert_close(standard, torch.tensor([9.1, 1.0]))
    torch.testing.assert_close(double, torch.tensor([1.9, 1.0]))
    assert not double.requires_grad and target.grad is None
    assert bootstrap_target(1.0, 9.0, False, 0.9) == 9.1
    assert bootstrap_target(1.0, 9.0, True, 0.9) == 1.0


def test_quantile_loss_hand_oracle_and_target_is_detached():
    predicted = torch.tensor([[0.0, 2.0]], requires_grad=True)
    target = torch.tensor([[1.0, 3.0]], requires_grad=True)
    # Midpoint levels .25/.75, errors [[1,3],[-1,1]]: weighted Huber sum=1.25.
    loss = quantile_huber(predicted, target)
    torch.testing.assert_close(loss, torch.tensor([1.25 / 4]))
    loss.sum().backward()
    assert target.grad is None and torch.isfinite(predicted.grad).all()


def test_prioritized_sampling_and_duplicate_updates():
    replay = Replay(3, seed=42, alpha=1.0)
    for i in range(3):
        replay.add(i, i % 2, float(i), i, False)
    replay.update([0, 1, 1, 2], [1.0, 2.0, 4.0, 8.0])
    np.testing.assert_allclose(replay.priority, [1.000001, 4.000001, 8.000001])
    ids, weights = replay.sample(30000, beta=1.0)
    np.testing.assert_allclose(
        np.bincount(ids, minlength=3) / len(ids), [1 / 13, 4 / 13, 8 / 13], atol=0.012
    )
    selected = np.array([weights[ids == i][0] for i in range(3)])
    np.testing.assert_allclose(selected, [1.0, 1 / 4, 1 / 8], rtol=2e-6)
