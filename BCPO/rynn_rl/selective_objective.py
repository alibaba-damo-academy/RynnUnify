"""BCPO group statistics. No gradients through reward targets."""

from dataclasses import dataclass
import torch


@dataclass
class Targets:
    advantages: torch.Tensor
    route_target: torch.Tensor
    route_mask: torch.Tensor
    direct_mean: torch.Tensor
    generate_mean: torch.Tensor


def build_targets(rewards, branches, valid, routable, temperature=0.1, cost=0.0, eps=1e-6):
    """Inputs [Q,8], branch 0=D/1=G; native questions must contain eight D.

    Group normalization uses population standard deviation.
    Invalid model protocols retain zero-reward body trajectories. Infrastructure
    failures must raise upstream and never become a reward in this function.
    """
    rewards = torch.as_tensor(rewards, dtype=torch.float32).detach()
    branches = torch.as_tensor(branches, dtype=torch.long, device=rewards.device)
    valid = torch.as_tensor(valid, dtype=torch.bool, device=rewards.device)
    routable = torch.as_tensor(routable, dtype=torch.bool, device=rewards.device)
    if rewards.ndim != 2 or rewards.shape[1] != 8 or not rewards.shape[0]:
        raise ValueError("Expected a nonempty [questions,8] batch")
    if branches.shape != rewards.shape or valid.shape != rewards.shape or routable.shape != rewards.shape[:1]:
        raise ValueError("Mismatched group shapes")
    if not temperature > 0 or not torch.isfinite(rewards).all() or ((rewards < 0) | (rewards > 1)).any():
        raise ValueError("Rewards must be finite in [0,1]; temperature must be positive")
    if ((branches != 0) & (branches != 1)).any():
        raise ValueError("Unknown branch")
    count_g = (branches == 1).sum(-1)
    if ((count_g != 4) & routable).any() or ((count_g != 0) & ~routable).any():
        raise ValueError("Expected 4D+4G for routable questions and 8D for native questions")
    advantages = torch.zeros_like(rewards)
    means = []
    for z in (0, 1):
        mask = branches == z
        count = mask.sum(-1).clamp_min(1)
        mean = (rewards * mask).sum(-1) / count
        variance = ((rewards - mean[:, None]).square() * mask).sum(-1) / count
        advantages += ((rewards - mean[:, None]) / (variance.sqrt()[:, None] + eps)) * mask
        means.append(mean)
    md, mg = means
    target = torch.sigmoid((mg - md - cost) / temperature).detach()
    mask = routable & valid.all(-1) & ((md > 0) | (mg > 0))
    return Targets(advantages.detach(), target, mask, md, mg)
