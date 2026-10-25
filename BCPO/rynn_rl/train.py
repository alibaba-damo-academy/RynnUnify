"""Portable, synchronous BCPO training. See docs/training.md."""

import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import torch
from .data import load_questions
from .hf_policy import HFPolicy
from .rewards import Reward
from .services import Generator, Judge
from .selective_rollout import Collector
from .selective_loss import LossConfig
from .trainer import update


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True, help="New output directory")
    p.add_argument("--generator-url", help="Existing /generate_video service")
    p.add_argument("--judge-url", help="Existing OpenAI-compatible base URL ending /v1")
    p.add_argument("--judge-model")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--batch-size", type=int, default=64, help="Questions per optimizer step; 8 trajectories each")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-context", type=int, default=32768)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--concurrency", type=int, default=4)
    return p


async def run(args):
    if min(args.batch_size, args.epochs, args.save_every, args.concurrency, args.max_new_tokens, args.max_context) < 1:
        raise ValueError("Counts must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    questions = load_questions(args.data)
    if any(q.routable for q in questions) and not args.generator_url:
        raise ValueError("Routable questions require --generator-url")
    if any(q.routable and not (q.images or q.videos) for q in questions):
        raise ValueError("Routable questions require observed media")
    if any(q.task in {"text", "mess"} for q in questions) and not (args.judge_url and args.judge_model):
        raise ValueError("Text/Mess data require --judge-url and --judge-model")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = LossConfig()
    (output / "config.json").write_text(json.dumps({**vars(args), "loss": asdict(config)}, indent=2) + "\n")
    policy = HFPolicy(args.model, args.device, args.dtype, args.max_new_tokens, args.max_context, not args.text_only)
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    judge = Judge(args.judge_url, args.judge_model) if args.judge_url else None
    generator = (
        Generator(args.generator_url, output / "videos", concurrency=args.concurrency) if args.generator_url else None
    )
    collector = Collector(policy, generator, Reward(judge), args.concurrency)
    step = 0
    for epoch in range(args.epochs):
        indices = list(range(len(questions)))
        random.shuffle(indices)
        # Same question count in final batch; explicit cyclic padding, no reward filtering.
        n = math.ceil(len(indices) / args.batch_size) * args.batch_size
        indices = [indices[i % len(indices)] for i in range(n)]
        for start in range(0, n, args.batch_size):
            batch = [questions[i] for i in indices[start : start + args.batch_size]]
            groups = await collector.collect(batch)
            metrics = update(policy, optimizer, batch, groups, config)
            step += 1
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "epoch": epoch, **metrics}, allow_nan=False) + "\n")
            traces = [
                {
                    "question_id": t.question_id,
                    "branch": t.branch,
                    "reward": t.reward,
                    "valid": t.valid,
                    "video": t.video,
                    "responses": [turn.text for turn in t.turns],
                }
                for group in groups
                for t in group
            ]
            (output / f"rollout-{step:06d}.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"step": step, **metrics}), flush=True)
            if step % args.save_every == 0:
                policy.save(output / f"checkpoint-{step:06d}")
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    policy.save(output / "final")
    # Model exports are interoperable HF checkpoints, not optimizer-resume snapshots.


def main():
    asyncio.run(run(parser().parse_args()))


if __name__ == "__main__":
    main()
