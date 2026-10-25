# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Episode schema, run configuration record, and duration metrics — all dependency-free.

Separated from the entry point so the parts that decide *what a valid run looks like*
and *what gets written into ``summary.json``* are testable without torch, CUDA, or a
trained checkpoint. The rollout itself lives in :mod:`interleave_vp.eval_interleave_plan`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from interleave_vp.duration_label import duration_label_seconds, snap_frames_4n1
from interleave_vp.text_parsing import is_done_instruction

STOP_KEYWORDS = ("completed", "finished", "done", "carried out", "wrapped up")
DEFAULT_MAX_PLAN_STEPS = 20


def normalize_episode(raw: dict) -> dict:
    """Validate the raw-episode schema and sort ``sub_plan`` by ``plan_index``.

    Fails loudly and specifically on a malformed record: a silently dropped
    ``first_frame_path`` or a duplicated ``plan_index`` would otherwise surface
    as a confusing crash mid-rollout, after the model is already loaded.
    """
    if not isinstance(raw, dict):
        raise ValueError("episode must be an object")
    try:
        sample_id = raw["id"]
        task = raw["high_level_task"]
        plans = raw["sub_plan"]
    except KeyError as exc:
        raise ValueError(f"episode is missing required field {exc.args[0]!r}") from exc
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("episode 'id' must be a non-empty string")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("episode 'high_level_task' must be a non-empty string")
    if not isinstance(plans, list) or not plans:
        raise ValueError("episode 'sub_plan' must be a non-empty list")

    seen: set[int] = set()
    for position, plan in enumerate(plans):
        label = f"sub_plan[{position}]"
        if not isinstance(plan, dict):
            raise ValueError(f"{label} must be an object")
        try:
            plan_index = plan["plan_index"]
            instruction = plan["instruction"]
            first_frame_path = plan["first_frame_path"]
        except KeyError as exc:
            raise ValueError(f"{label} is missing required field {exc.args[0]!r}") from exc
        if isinstance(plan_index, bool) or not isinstance(plan_index, int):
            raise ValueError(f"{label}.plan_index must be an integer")
        if plan_index in seen:
            raise ValueError(f"duplicate plan_index {plan_index} in sub_plan")
        if not isinstance(instruction, str):
            raise ValueError(f"{label}.instruction must be a string")
        if not isinstance(first_frame_path, str) or not first_frame_path.strip():
            raise ValueError(f"{label}.first_frame_path must be a non-empty string")
        seen.add(plan_index)

    episode = dict(raw)
    episode["sub_plan"] = sorted(plans, key=lambda plan: plan["plan_index"])
    return episode



RECORDED_PATH_FIELDS = ("first_frame_path", "clip_video_path", "target_frame_path")


def resolve_record_paths(episode: dict, base_dir: Path) -> dict:
    """Rewrite relative paths in the record against the JSONL's own directory.

    ``Image.open`` and ``cv2.VideoCapture`` resolve against the process working directory,
    so a dataset moved between machines would otherwise read fine until every frame of the
    rollout failed to load. Anchoring to the record file's directory makes the dataset
    relocatable and CWD-independent.
    """
    resolved = dict(episode)
    steps = []
    for plan in episode.get("sub_plan", []):
        step = dict(plan)
        for field in RECORDED_PATH_FIELDS:
            value = step.get(field)
            if isinstance(value, str) and value and not Path(value).is_absolute():
                step[field] = str(Path(base_dir) / value)
        steps.append(step)
    resolved["sub_plan"] = steps
    return resolved


def evaluation_config(args) -> dict:
    """The serialisable knob set shared by the run summary and every episode summary.

    This is the evidence a finished run carries about the recipe it actually used:
    a memory knob that silently failed to reach the CLI is visible here.
    """
    return {
        "resolution": args.resolution,
        "max_plan_steps": args.max_plan_steps,
        "history_window": args.history_window,
        "data_seed": args.data_seed,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "num_steps": args.num_steps,
        "guidance": args.guidance,
        "shift": args.shift,
        "seed": args.seed,
        "context_frames": args.context_frames,
        "context_fill": args.context_fill,
        "anchor_resolution": args.anchor_resolution,
        "use_gt_plan": args.use_gt_plan,
        "gt_text_and_length": args.gt_text_and_length,
        "use_pred_duration": args.use_pred_duration,
        "pred_duration_offset_s": args.pred_duration_offset,
        "reasoner_prompt_format": args.reasoner_prompt_format,
        "reasoner_input_size": (
            f"{args.reasoner_input_size[0]}x{args.reasoner_input_size[1]}"
            if args.reasoner_input_size is not None
            else None
        ),
        "generator_world_modeling": args.generator_world_modeling,
        "cross_chunk_v2v": args.cross_chunk_v2v,
        "cross_chunk_latents": args.cross_chunk_latents,
    }


def resolve_generation_frames(plan: dict, default_frames: int, use_gt_length: bool) -> tuple[int, int]:
    """``(requested, actual 4n+1)`` frame count for one step.

    Teacher forcing uses the labelled clip length; the VAE grid then floors it, so
    38 -> 37 and 63 -> 61.
    """
    requested = int(default_frames)
    if use_gt_length:
        if "frame_count" not in plan:
            raise ValueError("--gt-text-and-length requires frame_count on every sub_plan")
        gt_frames = int(plan["frame_count"])
        if gt_frames <= 0:
            raise ValueError(f"frame_count must be positive, got {gt_frames}")
        requested = max(5, min(gt_frames, 241))
    return requested, snap_frames_4n1(requested)


def is_task_completed(prediction: str) -> bool:
    """Closed-loop stop signal: the predicted plan says the task is done."""
    text_lower = prediction.lower()
    return any(keyword in text_lower for keyword in STOP_KEYWORDS)


def gt_duration_label(plan: dict) -> int | None:
    """GT whole-second label for a sub_plan, quantized to the training duration format.

    Terminal steps label 0; ``frame_count`` missing means the metric is skipped,
    not that the run failed.
    """
    if is_done_instruction(plan.get("instruction", "")):
        return 0
    frame_count = plan.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        return None
    return duration_label_seconds(frame_count, int(plan.get("fps", 24)))


def parse_duration_json(text: str) -> tuple[str, int] | None:
    """Parse ``{"action": ..., "duration": <int>}``, tolerating fences and chatter.

    Returns ``None`` on any shape failure so the caller falls back to the raw text
    and the default frame count. A boolean duration is rejected: ``True`` would
    silently become ``1`` second.
    """
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    action = obj.get("action")
    duration = obj.get("duration")
    if not isinstance(action, str) or not action.strip():
        return None
    if isinstance(duration, bool) or not isinstance(duration, int):
        return None
    return action.strip(), duration


def build_duration_comparison(plan: dict, predicted_plan: str, parsed: tuple[str, int] | None) -> dict:
    """The per-step GT-vs-prediction record written to ``comparison.json``."""
    frame_count = plan.get("frame_count")
    valid = isinstance(frame_count, int) and not isinstance(frame_count, bool) and frame_count > 0
    gt_duration = gt_duration_label(plan)
    pred_action, pred_duration = parsed if parsed is not None else (None, None)
    comparison = {
        "gt": {
            "action": plan["instruction"],
            "frame_count": frame_count,
            "fps": plan.get("fps", 24),
            "snapped_frame_count": snap_frames_4n1(frame_count) if valid else None,
            "duration_s": gt_duration,
        },
        "prediction": {
            "raw": predicted_plan,
            "parse_ok": parsed is not None,
            "action": pred_action,
            "duration_s": pred_duration,
        },
        "comparison": {
            "action_exact": pred_action == plan["instruction"] if pred_action is not None else None,
            "duration_error_s": None,
            "duration_abs_error_s": None,
            "duration_exact": None,
            "duration_within_1s": None,
        },
    }
    if gt_duration is not None and pred_duration is not None:
        error = pred_duration - gt_duration
        comparison["comparison"].update(
            {
                "duration_error_s": error,
                "duration_abs_error_s": abs(error),
                "duration_exact": error == 0,
                "duration_within_1s": abs(error) <= 1,
            }
        )
    return comparison


def build_episode_summary(episode: dict, results: list[dict], args) -> dict:
    """Episode summary, rewritten after every step so a crash keeps the prefix."""
    summary = {
        "episode_id": episode["id"],
        "data_source": episode.get("data_source", "unknown"),
        "high_level_task": episode["high_level_task"],
        "num_plans_in_data": len(episode["sub_plan"]),
        "num_steps_executed": len(results),
        "stopped_early": any("stopped_reason" in s for s in results),
        "config": evaluation_config(args),
        "gt_plans": [clip["instruction"] for clip in episode["sub_plan"]],
        "predicted_plans": [s["predicted_plan"] for s in results],
        "steps": results,
    }
    if getattr(args, "use_pred_duration", False) or getattr(args, "reasoner_prompt_format", "bare") == "tags":
        summary["duration_parse_failures"] = sum(1 for s in results if s.get("duration_parse_ok") is False)
        summary["tags_format_failures"] = sum(1 for s in results if s.get("tags_format_ok") is False)
    return summary


def duration_metrics(all_results: list[dict]) -> dict:
    """Aggregate duration-prediction quality over episode summaries.

    Rates are ``None`` when there is nothing to divide by — a run without
    ``frame_count`` labels reports ``compared: 0`` rather than a fake accuracy.
    """
    total = parsed = compared = exact = within1 = 0
    pred_hist: dict[int, int] = {}
    gt_hist: dict[int, int] = {}
    for episode in all_results:
        for step in episode.get("steps", []):
            if "duration_parse_ok" not in step:
                continue
            total += 1
            pred = step.get("pred_duration_s")
            gt = step.get("gt_duration_s")
            if step["duration_parse_ok"] and pred is not None:
                parsed += 1
                pred_hist[pred] = pred_hist.get(pred, 0) + 1
            if gt is not None:
                gt_hist[gt] = gt_hist.get(gt, 0) + 1
                if pred is not None:
                    compared += 1
                    exact += int(pred == gt)
                    within1 += int(abs(pred - gt) <= 1)
    return {
        "steps": total,
        "parse_failures": total - parsed,
        "parse_rate": (parsed / total) if total else None,
        "compared": compared,
        "exact_match": (exact / compared) if compared else None,
        "within_1s": (within1 / compared) if compared else None,
        "pred_duration_hist": {str(k): v for k, v in sorted(pred_hist.items())},
        "gt_duration_hist": {str(k): v for k, v in sorted(gt_hist.items())},
    }
