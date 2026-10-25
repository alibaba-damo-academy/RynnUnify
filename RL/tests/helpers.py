"""Test fixtures for gradient and rollout regression checks."""

import copy
from types import SimpleNamespace
import torch
from rynn_rl.data import Question
from rynn_rl.hf_policy import Turn, token_statistics
from rynn_rl.rewards import Reward
from rynn_rl.selective_rollout import Collector
from rynn_rl.trainer import update


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 12)
        self.recurrent = torch.nn.GRU(12, 12, batch_first=True)
        self.head = torch.nn.Linear(12, 8)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden, _ = self.recurrent(self.embedding(input_ids))
        return SimpleNamespace(logits=self.head(hidden))


class FixturePolicy:
    """Synthetic scripted text wrapping sampled categorical actions; TEST ONLY."""

    def __init__(self):
        self.model = TinyModel()
        self.reference = copy.deepcopy(self.model).requires_grad_(False)
        self.count = 0

    async def sample(self, messages, route=None):
        self.count += 1
        inputs = {"input_ids": torch.tensor([[1, 2 if route == "D" else 3]])}
        # Alternation ensures both high and low reward cases in a small regression test.
        token = torch.tensor([4 + self.count % 2])
        if route == "G":
            text = "<world_modeling>Continue the visible interaction.</world_modeling><duration>4s</duration>"
        else:
            text = (
                "<thinking>Fixture reasoning.</thinking><answer>" + ("A" if int(token[0]) == 4 else "B") + "</answer>"
            )
        turn = Turn(inputs, token, text, True, torch.empty(0))
        with torch.no_grad():
            turn.old_log_probs = token_statistics(self.model, turn)[0]
        return turn

    def route_logits(self, messages):
        return self.model(torch.tensor([[1, 2]])).logits[0, -1, [4, 5]]


async def exercise():
    torch.manual_seed(7)
    policy = FixturePolicy()
    before = [x.detach().clone() for x in policy.model.parameters()]
    reference = [x.detach().clone() for x in policy.reference.parameters()]
    question = Question("synthetic", "Which option is correct?", "A", "choice", images=["fixture.png"])

    async def generator(*args):
        return "synthetic-video-fixture.mp4"  # Explicit test double, never used by train.py.

    collector = Collector(policy, generator, Reward(), concurrency=1)
    groups = await collector.collect([question])
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=1e-3)
    metrics = update(policy, optimizer, [question], groups)
    changed = any(not torch.equal(a, b) for a, b in zip(before, policy.model.parameters()))
    frozen = all(torch.equal(a, b) for a, b in zip(reference, policy.reference.parameters()))
    assert changed and frozen and len(groups[0]) == 8
    assert all(len(t.turns) == (2 if t.branch else 1) for t in groups[0])
    return {
        "fixture_only": True,
        "actor_updated": changed,
        "reference_unchanged": frozen,
        "trajectories": 8,
        "metrics": metrics,
    }

