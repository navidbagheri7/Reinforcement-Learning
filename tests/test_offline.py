import torch

from decision_lab.offline import conservative_penalty, expectile_loss


def test_conservative_gradient_and_illegal_action():
    q = torch.tensor([[0.0, 1.0, 999.0]], requires_grad=True)
    loss = conservative_penalty(q, torch.tensor([0]), torch.tensor([[True, True, False]])).sum()
    loss.backward()
    assert torch.allclose(q.grad, torch.tensor([[-0.7310586, 0.7310586, 0.0]]), atol=1e-6)


def test_expectile_oracle_and_detached_target():
    target = torch.tensor([0.0, 2.0], requires_grad=True)
    v = torch.tensor([1.5], requires_grad=True)
    loss = expectile_loss(v, target, 0.75).mean()
    loss.backward()
    assert abs(float(v.grad)) < 1e-7
    assert target.grad is None


def test_twin_critics_are_independent_and_targets_frozen():
    from decision_lab.offline import independent_critics

    torch.manual_seed(42)
    q, q2, tq, tq2 = independent_critics({"features": 6, "actions": 4, "width": 16}, "cpu")
    x = torch.ones((4, 6))
    assert not torch.equal(q(x), q2(x))
    assert torch.equal(q(x), tq(x)) and torch.equal(q2(x), tq2(x))
    assert all(not p.requires_grad for target in [tq, tq2] for p in target.parameters())


def test_logged_batch_reuse_keeps_bytes(tmp_path):
    import numpy as np

    from decision_lab.artifacts import sha256
    from decision_lab.offline import save_logged

    path = tmp_path / "logged.npz"
    batch = {"state": np.array([0, 1]), "action": np.array([0, 0])}
    counts = np.array([[1], [1]])
    save_logged(path, batch, counts)
    before = sha256(path)
    save_logged(path, batch, counts)
    assert sha256(path) == before
