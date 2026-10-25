import asyncio
import pytest
from rynn_rl.data import Question
from rynn_rl.protocol import messages, request, answer
from tests.helpers import FixturePolicy, exercise
from rynn_rl.selective_rollout import Collector
from rynn_rl.rewards import Reward, FeedbackError, phys_error
from rynn_rl.legacy.sharp_reward import curve


def test_real_gradient_update_offline():
    report = asyncio.run(exercise())
    assert report["actor_updated"] and report["reference_unchanged"]


def test_ground_truth_never_in_prompt():
    q = Question("id", "question", "SECRET_REFERENCE", "choice", ["observed.png"], [])
    assert "SECRET_REFERENCE" not in repr(messages(q))
    assert "observed.png" in repr(messages(q))


def test_service_failure_not_zero_reward():
    async def broken(*args):
        raise FeedbackError("offline")

    q = Question("id", "question", "A", "choice", ["fixture.png"], [])
    collector = Collector(FixturePolicy(), broken, Reward(), 1)
    with pytest.raises(FeedbackError):
        asyncio.run(collector.collect([q]))


def test_protocols():
    assert request("<world_modeling>move</world_modeling><duration>4s</duration>") == ("move", 4.0)
    assert request("<world_modeling>move</world_modeling><duration>11s</duration>") is None
    assert request("<world_modeling>move</world_modeling><duration>4s</duration><answer>A</answer>") is None
    assert answer("<thinking>x</thinking><answer>A</answer>") == "A"
    assert answer("<thinking>x</thinking><answer>A</answer><answer>B</answer>") is None


def test_reward_units_sign_and_cutoff():
    assert phys_error("-1 m", "-100 cm") == 0
    assert phys_error("100 cm", "-100 cm") == 2
    assert phys_error("2 s", "2 cm") is None
    assert phys_error("0", "0 s") == 0
    assert phys_error("1", "0 s") is None
    assert curve(0.1) == pytest.approx((2 / 3) ** 3)
    assert curve(0.3) == 0


def test_geometry_rewards():
    async def run():
        reward = Reward()
        q = Question("p", "q", "(500,500)", "point")
        assert await reward(q, "(500,500)") == 1
        assert await reward(q, "(900,900)") == 0
        q = Question("t", "q", "(0,0),(500,500),(1000,1000)", "trajectory")
        assert await reward(q, q.reference) == 1
        assert await reward(q, "invalid") == 0

    asyncio.run(run())


def test_judge_failure_retries_same_answer():
    seen = []

    async def judge(msg):
        seen.append(msg)
        return "invalid"

    q = Question("x", "q", "reference", "text")
    with pytest.raises(FeedbackError):
        asyncio.run(Reward(judge)(q, "prediction"))
    assert len(seen) == 3 and seen[0] == seen[1] == seen[2]
