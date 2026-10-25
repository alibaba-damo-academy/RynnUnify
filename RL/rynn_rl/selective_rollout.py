"""Balanced D/G exploration and complete request-observe-answer trajectories."""

import asyncio
from dataclasses import dataclass, field
from . import protocol
from .legacy.standalone_scoring import extract_answer


@dataclass
class Trajectory:
    question_id: str
    branch: int
    turns: list = field(default_factory=list)
    reward: float = 0.0
    valid: bool = False
    video: str | None = None


class Collector:
    def __init__(self, policy, generator, reward, concurrency=4):
        self.policy, self.generator, self.reward = policy, generator, reward
        self.semaphore = asyncio.Semaphore(concurrency)

    async def one(self, question, branch):
        async with self.semaphore:
            route = "G" if branch else "D"
            history = protocol.messages(question)
            first = await self.policy.sample(history, route=route)
            trajectory = Trajectory(question.id, branch, [first])
            if branch:
                parsed = protocol.request(first.text)
                if parsed is None or not first.complete:
                    return trajectory
                prompt, duration = parsed
                # Invalid model image references are protocol errors, not outages.
                import re

                count = 1 if question.videos else len(question.images)
                refs = re.findall(r"<(?:frame|Picture)\s*([^<>]*)>", prompt, re.I)
                if any(not r.strip().isdigit() or not 1 <= int(r) <= count for r in refs):
                    return trajectory
                trajectory.video = await self.generator(question, prompt, duration)
                history += [
                    {"role": "assistant", "content": "<route>G</route>" + first.text},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": trajectory.video},
                            {"type": "text", "text": "Use this generated video to answer the original question."},
                        ],
                    },
                ]
                final = await self.policy.sample(history, route=None)
                trajectory.turns.append(final)
            else:
                final = first
            candidate = protocol.answer(final.text)
            trajectory.valid = candidate is not None and all(t.complete for t in trajectory.turns)
            if question.native:
                candidate = extract_answer(final.text)
                trajectory.reward = await self.reward(question, candidate)
            elif trajectory.valid:
                trajectory.reward = await self.reward(question, candidate)
            return trajectory

    async def collect(self, questions):
        # Whole batch is returned only after all feedback succeeds. No DAPO.
        jobs = [self.one(q, z) for q in questions for z in ([0] * 4 + [1] * 4 if q.routable else [0] * 8)]
        tasks = [asyncio.create_task(job) for job in jobs]
        try:
            flat = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return [flat[i : i + 8] for i in range(0, len(flat), 8)]
