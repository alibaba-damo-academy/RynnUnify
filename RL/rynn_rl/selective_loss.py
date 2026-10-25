"""Differentiable sequence GSPO and full-question-batch soft routing losses."""

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    clip_low: float = 0.0003
    clip_high: float = 0.0004
    route_weight: float = 0.01
    kl_weight: float = 0.001
    entropy_weight: float = 0.001


def body_loss(log_probs, old_log_probs, reference_log_probs, entropy, action_mask, advantages, config=LossConfig()):
    """Each row is a full trajectory; concatenate G request and answer actions.

    Prompt, forced route, tool output and padding have mask zero. An all-masked
    trajectory is an implementation error, never silently dropped. KL uses the
    nonnegative sampled k3 estimator; reference and behavior values are detached.
    """
    if any(t.shape != log_probs.shape for t in (old_log_probs, reference_log_probs, entropy, action_mask)):
        raise ValueError("Token tensors must have identical shapes")
    if log_probs.ndim != 2 or advantages.shape != log_probs.shape[:1]:
        raise ValueError("Expected token arrays [N,T] and advantages [N]")
    mask = action_mask.to(log_probs.dtype)
    if not ((mask == 0) | (mask == 1)).all() or (mask.sum(-1) == 0).any():
        raise ValueError("Every trajectory needs a nonempty binary action mask")

    def mean(x):
        return torch.where(mask.bool(), x, 0).sum(-1) / mask.sum(-1)

    log_ratio = mean(log_probs - old_log_probs.detach())
    ratio = log_ratio.exp()
    adv = advantages.detach()
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - config.clip_low, 1 + config.clip_high) * adv)
    delta = torch.where(mask.bool(), reference_log_probs.detach() - log_probs, 0)
    kl = mean(delta.exp() - delta - 1)
    ent = mean(entropy)
    loss = -surrogate.mean() + config.kl_weight * kl.mean() - config.entropy_weight * ent.mean()
    return loss, {
        "body": -surrogate.detach().mean(),
        "kl": kl.detach().mean(),
        "entropy": ent.detach().mean(),
        "ratio": ratio.detach().mean(),
    }


def route_loss(route_logits, targets, mask):
    """Two logits per question in D,G order; denominator includes ALL Q."""
    if route_logits.shape != (*targets.shape, 2) or mask.shape != targets.shape:
        raise ValueError("Expected logits [Q,2], targets/mask [Q]")
    difference = route_logits[..., 1] - route_logits[..., 0]
    per_question = F.binary_cross_entropy_with_logits(difference, targets.detach(), reduction="none")
    return (per_question * mask.to(per_question.dtype)).mean()
