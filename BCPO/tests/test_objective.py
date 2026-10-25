import pytest
import torch
from rynn_rl.selective_objective import build_targets
from rynn_rl.selective_loss import body_loss, route_loss, LossConfig


def test_branch_statistics_and_mask_denominator():
    rewards = torch.tensor([[0.2, 0.4, 0.6, 0.8, 0.6, 0.7, 0.8, 0.9], [0.0] * 8])
    targets = build_targets(rewards, [[0] * 4 + [1] * 4] * 2, [[True] * 8] * 2, [True, True])
    assert targets.advantages[0, :4].mean().abs() < 1e-6
    assert targets.advantages[0, 4:].mean().abs() < 1e-6
    assert targets.route_target[0].item() == pytest.approx(0.9241418)
    assert targets.route_mask.tolist() == [True, False]
    logits = torch.zeros(2, 2, requires_grad=True)
    loss = route_loss(logits, targets.route_target, targets.route_mask)
    assert loss.item() == pytest.approx(0.69314718 / 2)
    loss.backward()
    assert logits.grad[0, 1] < 0 and logits.grad[1].abs().sum() == 0


def test_invalid_protocol_disables_only_routing_and_native_8d():
    valid = [[True] * 8]
    valid[0][2] = False
    targets = build_targets([[0, 1, 0, 1, 0.1, 0.3, 0.5, 0.7]], [[0] * 4 + [1] * 4], valid, [True])
    assert not targets.route_mask[0] and targets.advantages.abs().sum() > 0
    native = build_targets([[0, 1] * 4], [[0] * 8], [[True] * 8], [False])
    assert not native.route_mask[0] and native.advantages[0].std(correction=0) > 0.99
    with pytest.raises(ValueError):
        build_targets([[0.0] * 8], [[0] * 8], [[True] * 8], [True])


def test_equal_within_branch_rewards_still_train_route():
    t = build_targets([[0.2] * 4 + [0.8] * 4], [[0] * 4 + [1] * 4], [[True] * 8], [True])
    assert torch.allclose(t.advantages, torch.zeros_like(t.advantages), atol=1e-6)
    assert t.route_target[0] > 0.99 and t.route_mask[0]


def test_gspo_uses_one_geometric_ratio_and_masks_tools():
    current = torch.tensor([[0.1, 0.3, 100.0]], requires_grad=True)
    old = torch.zeros_like(current)
    ref = current.detach().clone()
    loss, stats = body_loss(
        current,
        old,
        ref,
        torch.zeros_like(current),
        torch.tensor([[1, 1, 0]]),
        torch.ones(1),
        LossConfig(clip_low=0.9, clip_high=1.0, kl_weight=0, entropy_weight=0),
    )
    assert stats["ratio"].item() == pytest.approx(torch.exp(torch.tensor(0.2)).item())
    loss.backward()
    assert current.grad[0, 0] == current.grad[0, 1] and current.grad[0, 2] == 0


def test_asymmetric_clipping_and_detached_targets():
    current = torch.tensor([[0.1], [-0.1]], requires_grad=True)
    old = torch.zeros(2, 1, requires_grad=True)
    advantages = torch.tensor([1.0, -1.0], requires_grad=True)
    loss, _ = body_loss(
        current,
        old,
        current.detach(),
        torch.zeros(2, 1),
        torch.ones(2, 1),
        advantages,
        LossConfig(kl_weight=0, entropy_weight=0),
    )
    assert loss.item() == pytest.approx((-0.0004 - 0.0003) / 2, abs=1e-7)
    loss.backward()
    assert old.grad is None and advantages.grad is None
    assert current.grad.abs().sum() == 0


def test_rewards_nonfinite_rejected():
    with pytest.raises(ValueError):
        build_targets([[float("nan")] * 8], [[0] * 4 + [1] * 4], [[True] * 8], [True])
