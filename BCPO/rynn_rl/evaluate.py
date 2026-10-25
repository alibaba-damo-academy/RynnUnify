"""Evaluate learned routing, D-only or G-only using the same public protocol."""

import argparse
import asyncio
import json
from pathlib import Path
import torch
from .data import load_questions
from .hf_policy import HFPolicy
from .protocol import messages
from .rewards import Reward
from .selective_rollout import Collector
from .services import Generator, Judge


async def run(args):
    questions = load_questions(args.data)
    if args.mode != "direct" and any(q.routable for q in questions) and not args.generator_url:
        raise ValueError("Learned/generate modes require a generator")
    if any(q.task in {"text", "mess"} for q in questions) and not (args.judge_url and args.judge_model):
        raise ValueError("Text/Mess data require --judge-url and --judge-model")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    policy = HFPolicy(args.model, device=args.device, dtype=args.dtype, load_reference=False)
    # Evaluation uses greedy decoding, training samples unfiltered temperature 1.
    original_sample = policy.sample

    async def greedy(*a, **kw):
        return await original_sample(*a, **kw, greedy=True)

    policy.sample = greedy
    judge = Judge(args.judge_url, args.judge_model) if args.judge_url and args.judge_model else None
    generator = Generator(args.generator_url, out / "videos") if args.generator_url else None
    collector = Collector(policy, generator, Reward(judge), 1)
    records = []
    for question in questions:
        probability = None
        if not question.routable or args.mode == "direct":
            branch = 0
        elif args.mode == "generate":
            branch = 1
        else:
            with torch.no_grad():
                logits = policy.route_logits(messages(question))
                probability = float(logits.softmax(-1)[1])
                branch = int(logits.argmax())
        trajectory = await collector.one(question, branch)
        record = {
            "id": question.id,
            "branch": branch,
            "p_generate": probability,
            "reward": trajectory.reward,
            "valid": trajectory.valid,
            "video": trajectory.video,
            "responses": [t.text for t in trajectory.turns],
        }
        records.append(record)
        with (out / "predictions.jsonl").open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "questions": len(records),
        "mean_training_reward": sum(r["reward"] for r in records) / len(records),
        "generation_rate": sum(r["branch"] for r in records) / len(records),
        "mode": args.mode,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "data", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--mode", choices=["learned", "direct", "generate"], default="learned")
    p.add_argument("--generator-url")
    p.add_argument("--judge-url")
    p.add_argument("--judge-model")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
