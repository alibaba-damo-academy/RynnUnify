"""BCPO update; accumulate trajectory gradients with one optimizer step per Q."""

import torch
from .hf_policy import token_statistics
from .protocol import messages
from .selective_objective import build_targets
from .selective_loss import body_loss, route_loss, LossConfig


def update(policy, optimizer, questions, groups, config=LossConfig(), max_grad_norm=1.0):
    q = len(questions)
    if q == 0 or len(groups) != q:
        raise ValueError("Question/group alignment mismatch")
    if any(
        len(group) != 8 or any(t.question_id != question.id for t in group)
        for question, group in zip(questions, groups)
    ):
        raise ValueError("Each question needs its own eight trajectories")
    targets = build_targets(
        [[t.reward for t in g] for g in groups],
        [[t.branch for t in g] for g in groups],
        [[t.valid for t in g] for g in groups],
        [x.routable for x in questions],
    )
    device = next(policy.model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    metrics = {"body": 0.0, "kl": 0.0, "entropy": 0.0, "ratio": 0.0, "route": 0.0}
    try:
        for j, group in enumerate(groups):
            for i, trajectory in enumerate(group):
                current, previous, reference, entropies = [], [], [], []
                for turn in trajectory.turns:
                    logp, entropy = token_statistics(policy.model, turn)
                    with torch.no_grad():
                        ref, _ = token_statistics(policy.reference, turn)
                    current.append(logp)
                    previous.append(turn.old_log_probs.to(device))
                    reference.append(ref.to(device))
                    entropies.append(entropy)
                # Both G turns form ONE sequence ratio and receive one advantage.
                values = [torch.cat(parts)[None] for parts in (current, previous, reference, entropies)]
                loss, stats = body_loss(
                    *values, torch.ones_like(values[0]), targets.advantages[j, i].reshape(1).to(device), config
                )
                (loss / (q * 8)).backward()
                for key, value in stats.items():
                    metrics[key] += float(value) / (q * 8)
            # Only original observations appear at this route position.
            if targets.route_mask[j]:
                logits = policy.route_logits(messages(questions[j]))[None]
                route = (
                    route_loss(
                        logits, targets.route_target[j : j + 1].to(device), targets.route_mask[j : j + 1].to(device)
                    )
                    / q
                )
                (config.route_weight * route).backward()
                metrics["route"] += float(route.detach())
        norm = torch.nn.utils.clip_grad_norm_(policy.model.parameters(), max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    metrics.update(
        reward=sum(t.reward for g in groups for t in g) / (q * 8),
        route_valid_fraction=float(targets.route_mask.float().mean()),
        grad_norm=float(norm),
        total=metrics["body"]
        + config.kl_weight * metrics["kl"]
        - config.entropy_weight * metrics["entropy"]
        + config.route_weight * metrics["route"],
    )
    return metrics
