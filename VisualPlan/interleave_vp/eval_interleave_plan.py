#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Interleaved plan evaluation with scene memory.

One Cosmos3-Nano ``OmniMoTModel`` instance carries both branches: its ``und``
branch is the Reasoner (predicts the next sub-plan, its dense visual description
and a duration) and its ``gen`` branch is the Generator (renders that sub-plan as
one video chunk). The chunk's last frame becomes the next observation, so the
rollout is closed-loop.

Per step the Generator is conditioned on three things beyond the observation
frame, and each is a switch because each corresponds to a training recipe:

* scene-memory context items (``--context-frames`` / ``--context-fill`` /
  ``--anchor-resolution``),
* a clean V2V prefix continued from the previous chunk (``--cross-chunk-v2v`` /
  ``--cross-chunk-latents``),
* an optional goal frame for a labelled ``target_frame_path`` step.

Chunk length comes from the predicted duration (``--use-pred-duration``) or, for
the teacher-forced upper bound, from each clip's GT length (``--gt-text-and-length``).

Run it through ``run_interleave.sh``; the flags and their trained-recipe contract
are documented in README.md.

Usage:
    torchrun --nproc_per_node=8 -m interleave_vp.eval_interleave_plan \\
        --test-data episodes.jsonl --output-dir outputs/run1 --checkpoint-path <hf_export>
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosmos_framework.inference.common.init import init_script

init_script()

import torch.distributed as dist
from PIL import Image

from cosmos_framework.data.vfm.local_datasets.helper import get_aspect_ratio
from cosmos_framework.utils import log
from interleave_vp.cli_args import build_parser, validate_data_parallel_args, validate_run_args
from interleave_vp.context_fill import build_fill_plan, select_history_units
from interleave_vp.cross_chunk_v2v import combined_timeline, prefix_skip_reason
from interleave_vp.duration_label import frames_from_duration, snap_frames_4n1
from interleave_vp.episodes import (
    build_duration_comparison as _build_duration_comparison,
    build_episode_summary as _build_episode_summary,
    duration_metrics as _duration_metrics,
    evaluation_config as _evaluation_config,
    gt_duration_label as _gt_duration_label,
    is_task_completed as _is_task_completed,
    normalize_episode as _normalize_episode,
    resolve_record_paths as _resolve_record_paths,
    parse_duration_json as _parse_duration_json,
    resolve_generation_frames as _resolve_generation_frames,
)
from interleave_vp.merge_interleave_video import (
    DEFAULT_OUTPUT_NAME,
    MANIFEST_NAME,
    merge_episode_videos,
)
from interleave_vp.text_parsing import parse_duration_seconds, parse_interleaved_tag_prediction
from interleave_vp.utils import is_main_rank, load_test_data, save_results

# ── Model loading ───────────────────────────────────────────────────────────


def load_model(args):
    """Load OmniMoTModel via OmniInference (handles FSDP, parallelism, etc).

    Supports DCP and HuggingFace checkpoint formats.
    Config is loaded from --config-file (YAML) or auto-detected (HF config.json).
    """
    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.inference import OmniInference

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    overrides = OmniSetupOverrides(
        checkpoint_path=args.checkpoint_path,
        output_dir=Path(args.output_dir),
        sampler="unipc",
    )
    if args.config_file:
        overrides.config_file = args.config_file
    if args.experiment:
        # DCP + MODULE path: build the model from the training experiment config
        # so the architecture matches the checkpoint (the inference YAML may drift).
        overrides.experiment = args.experiment
        if not args.config_file:
            overrides.config_file = "cosmos_framework/configs/base/config.py"
        if args.experiment_override:
            overrides.experiment_overrides = list(args.experiment_override)

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if args.data_parallel:
        overrides.dp_shard_size = args.dp_shard_size
        overrides.dp_replicate_size = args.dp_replicate_size or world_size
        overrides.cp_size = args.cp_size
        overrides.cfgp_size = args.cfgp_size

    setup_args = overrides.build_setup(world_size=world_size, local_world_size=local_world_size)

    inference = OmniInference._create(setup_args, guardrails=None, _timer=None)
    return inference.model



def evaluate_episode(
    episode: dict,
    reasoner,
    generator,
    output_dir: Path,
    args,
) -> list[dict]:
    write_outputs = args.data_parallel or is_main_rank()
    sample_id = episode["id"]
    task = episode["high_level_task"]
    plans = episode["sub_plan"]
    initial_frame = Image.open(plans[0]["first_frame_path"]).convert("RGB")
    current_obs = initial_frame
    history: list[tuple[str, Image.Image]] = []
    results: list[dict] = []

    initial_ar = get_aspect_ratio(initial_frame.size[0], initial_frame.size[1])

    num_plans = len(plans)
    if args.max_plan_steps is not None:
        num_plans = min(num_plans, args.max_plan_steps)

    summary_path = output_dir / "summary.json"
    gen_keyframes: list[Image.Image] = []
    # --context-fill: full frames of the EARLIEST generated chunks, needed to
    # top up the context on the first steps. Filling stops once the history can
    # supply K-1 distinct chunks, so at most context_frames-2 chunks are ever
    # retained — 2 for the usual K=4.
    ctx_fill_chunks: list[list[Image.Image]] = []
    # Cross-chunk V2V: full frames of the immediately-previous generated
    # chunk, sliced into the current step's V2V prefix.
    prev_chunk_frames: list[Image.Image] | None = None

    for step_idx in range(num_plans):
        plan = plans[step_idx]
        gt_plan = plan["instruction"]
        step_dir = output_dir / f"step_{step_idx:03d}"
        requested_num_frames, generated_num_frames = _resolve_generation_frames(
            plan, args.num_frames, args.gt_text_and_length
        )
        if args.use_pred_duration:
            log.info(
                f"  [{sample_id}] starting step {step_idx}/{num_plans-1}: "
                f"gt_frames={plan.get('frame_count')} | "
                f"generator_frames=<pred-duration> | "
                f"gt_text={gt_plan!r}"
            )
        else:
            log.info(
                f"  [{sample_id}] starting step {step_idx}/{num_plans-1}: "
                f"gt_frames={plan.get('frame_count')} | "
                f"generator_frames={requested_num_frames}->{generated_num_frames} | "
                f"gt_text={gt_plan!r}"
            )


        # The Generator always conditions on the previous chunk's last frame;
        # that closed-loop conditioning is exactly what this rollout evaluates.
        generator_obs = current_obs

        t0 = time.time()
        if args.use_gt_plan or args.gt_text_and_length:
            # This combined checkpoint's reasoner LM is text-only (no visual
            # module), so image-conditioned prediction is unavailable; in GT
            # mode the generator is driven by the labelled plans anyway.
            predicted_plan = gt_plan
        else:
            predicted_plan = reasoner.predict_next_step(
                task,
                initial_frame,
                history[-args.history_window:],
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
                with_duration=args.use_pred_duration,
                prompt_format=args.reasoner_prompt_format,
                current_obs=generator_obs,
                # Data-parallel ranks run unrelated episodes and therefore do
                # not enter token decoding in lockstep. Model-parallel ranks
                # still synchronize EOS to preserve FSDP collectives.
                sync_eos_across_ranks=not args.data_parallel,
            )
        reasoner_time = time.time() - t0

        # Parse the prediction. 'tags' mode independently extracts the
        # training-native <subplan>/<world_modeling>/<duration> fields. Missing
        # subplan/world-modeling fields retain the complete raw completion for
        # that field; a missing duration alone falls back to default frames.
        # With --use-pred-duration, the parsed <duration>, plus the optional
        # --pred-duration-offset, drives the Generator window.
        # 'bare' + --use-pred-duration: the historical JSON {"action","duration"}.
        pred_action = predicted_plan
        pred_duration: float | None = None
        pred_world_modeling: str | None = None
        tags_parse_ok: bool | None = None
        subplan_tag_parse_ok: bool | None = None
        world_modeling_tag_parse_ok: bool | None = None
        duration_tag_parse_ok: bool | None = None
        duration_comparison: dict | None = None
        if args.reasoner_prompt_format == "tags":
            parsed_tags = parse_interleaved_tag_prediction(predicted_plan)
            tags_parse_ok = bool(parsed_tags["tags_format_ok"])
            subplan_tag_parse_ok = bool(parsed_tags["subplan_tag_parse_ok"])
            world_modeling_tag_parse_ok = bool(parsed_tags["world_modeling_tag_parse_ok"])
            duration_tag_parse_ok = bool(parsed_tags["duration_tag_parse_ok"])
            pred_action = str(parsed_tags["subplan"])
            pred_world_modeling = str(parsed_tags["world_modeling"])
            pred_duration = parse_duration_seconds(parsed_tags.get("duration"))
            gt_duration = _gt_duration_label(plan)
            if pred_duration is not None:
                if args.use_pred_duration:
                    generator_duration = pred_duration + args.pred_duration_offset
                    requested_num_frames = frames_from_duration(generator_duration, fps=args.fps)
                    generated_num_frames = requested_num_frames
                else:
                    generator_duration = None
                if gt_duration is not None:
                    error = round(pred_duration - gt_duration, 3)
                    duration_comparison = {
                        "raw": predicted_plan,
                        "pred_duration_s": pred_duration,
                        "gt_duration_s": gt_duration,
                        "duration_error_s": error,
                        "duration_abs_error_s": abs(error),
                        "duration_exact": error == 0,
                        "duration_within_1s": abs(error) <= 1,
                    }
                    if args.use_pred_duration:
                        duration_comparison["pred_duration_offset_s"] = args.pred_duration_offset
                        duration_comparison["generator_duration_s"] = generator_duration
                log.info(
                    f"  [{sample_id}] step {step_idx}: tags subplan={pred_action[:60]!r} "
                    f"duration={pred_duration}s + offset={args.pred_duration_offset}s "
                    f"-> generator_duration={generator_duration}s "
                    f"-> generator_frames={generated_num_frames}"
                )
            else:
                log.info(
                    f"  [{sample_id}] step {step_idx}: <duration> parse failed, "
                    f"falling back to num_frames={generated_num_frames}; raw={predicted_plan[:120]!r}"
                )
        elif args.use_pred_duration:
            parsed = _parse_duration_json(predicted_plan)
            duration_comparison = _build_duration_comparison(plan, predicted_plan, parsed)
            if parsed is not None:
                pred_action, pred_duration = parsed
                requested_num_frames = frames_from_duration(pred_duration, fps=args.fps)
                generated_num_frames = requested_num_frames
                log.info(
                    f"  [{sample_id}] step {step_idx}: pred_duration={pred_duration}s -> "
                    f"generator_frames={generated_num_frames}"
                )
            else:
                log.info(
                    f"  [{sample_id}] step {step_idx}: duration JSON parse failed, "
                    f"falling back to num_frames={generated_num_frames}; raw={predicted_plan[:120]!r}"
                )

        if write_outputs:
            step_dir.mkdir(parents=True, exist_ok=True)
            (step_dir / "predicted_plan.txt").write_text(predicted_plan)
            (step_dir / "gt_plan.txt").write_text(gt_plan)
            if duration_comparison is not None:
                (step_dir / "comparison.json").write_text(
                    json.dumps(duration_comparison, indent=2, ensure_ascii=False)
                )
            generator_obs.save(str(step_dir / "current_obs.jpg"))

        step_result = {
            "step": step_idx,
            "plan_index": plan["plan_index"],
            "gt_plan": gt_plan,
            "predicted_plan": predicted_plan,
            "reasoner_time_s": round(reasoner_time, 2),
        }
        if args.reasoner_prompt_format == "tags":
            step_result["tags_format_ok"] = tags_parse_ok
            step_result["subplan_tag_parse_ok"] = subplan_tag_parse_ok
            step_result["world_modeling_tag_parse_ok"] = world_modeling_tag_parse_ok
            step_result["duration_tag_parse_ok"] = duration_tag_parse_ok
            step_result["duration_parse_ok"] = pred_duration is not None
            step_result["predicted_action"] = pred_action
            step_result["predicted_world_modeling"] = pred_world_modeling
            if pred_duration is not None:
                step_result["pred_duration_s"] = pred_duration
                if args.use_pred_duration:
                    step_result["pred_duration_offset_s"] = args.pred_duration_offset
                    step_result["generator_duration_s"] = generator_duration
            gt_duration = _gt_duration_label(plan)
            if gt_duration is not None:
                step_result["gt_duration_s"] = gt_duration
        elif args.use_pred_duration:
            step_result["duration_parse_ok"] = pred_duration is not None
            if pred_duration is not None:
                step_result["pred_duration_s"] = pred_duration
            gt_duration = _gt_duration_label(plan)
            if gt_duration is not None:
                step_result["gt_duration_s"] = gt_duration

        # In teacher-forced GT text+length mode, retain the Reasoner's completion
        # prediction for analysis but execute every labelled clip. Otherwise keep
        # the historical closed-loop early stop.
        predicted_task_completed = _is_task_completed(
            pred_action if args.reasoner_prompt_format == "tags" else predicted_plan
        )
        step_result["reasoner_predicted_task_completed"] = predicted_task_completed
        if predicted_task_completed and not args.gt_text_and_length:
            step_result["stopped_reason"] = "task_completed"
            results.append(step_result)
            log.info(
                f"  [{sample_id}] step {step_idx}: task completed — "
                f"pred='{predicted_plan[:80]}...'"
            )
            if write_outputs:
                with open(summary_path, "w") as f:
                    json.dump(_build_episode_summary(episode, results, args), f, indent=2, ensure_ascii=False)
            break

        t0 = time.time()
        if args.use_gt_plan or args.gt_text_and_length:
            generator_plan = gt_plan
        elif args.reasoner_prompt_format == "tags" or args.use_pred_duration:
            generator_plan = pred_action
        else:
            generator_plan = predicted_plan
        gen_temporal_caption = (
            pred_world_modeling
            if args.generator_world_modeling and args.reasoner_prompt_format == "tags"
            else None
        )
        step_result.update(
            {
                "generator_plan": generator_plan,
                "generator_temporal_caption": gen_temporal_caption,
                "gt_frame_count": plan.get("frame_count"),
                "requested_num_frames": requested_num_frames,
                "generated_frames": generated_num_frames,
            }
        )
        # Scene-memory context frames: episode first frame + the most
        # recent generated keyframes, mirroring the --context-frames training
        # composition (episode first frame + recent clips' first frames).
        ctx_images = None
        if args.context_frames > 0 and args.context_fill:
            # Follow the trained context layout: the initial frame is pinned, and the
            # rest are generated chunks chosen by the shared rule (evenly spread
            # earlier chunks + the most recent one). Keys are (chunk, frame) so
            # fill frames slot in ahead of their chunk's keyframe. Note the chunk
            # keyframe is a chunk's LAST frame here vs a clip's FIRST frame in
            # training — a pre-existing one-step offset in this recipe.
            k = len(gen_keyframes)
            if k == 0:
                ctx_images = [initial_frame] * args.context_frames
            else:
                units = select_history_units(k, args.context_frames)
                END = 1 << 30  # sorts a chunk's keyframe after its fill frames
                entries = [(u, END, gen_keyframes[u]) for u in units]
                n_fill = args.context_frames - 1 - len(units)
                if n_fill > 0 and ctx_fill_chunks:
                    retained = list(range(min(len(ctx_fill_chunks), k)))
                    fill_plan = build_fill_plan([len(ctx_fill_chunks[u]) for u in retained], n_fill)
                    for u, frame_idx in fill_plan:
                        chunk = retained[u]
                        entries.append((chunk, frame_idx, ctx_fill_chunks[chunk][frame_idx]))
                entries.sort(key=lambda e: (e[0], e[1]))
                ctx_images = [initial_frame] + [e[2] for e in entries]
        elif args.context_frames > 0:
            ctx_images = [initial_frame] + gen_keyframes[-(args.context_frames - 1) :]
        need_ctx_frames = (
            args.context_fill
            and args.context_frames > 0
            and len(ctx_fill_chunks) < args.context_frames - 2
        )
        # The action window is what the current step would generate WITHOUT a
        # prefix: the GT/predicted length when forced, else the generator default.
        forced_len = args.gt_text_and_length or (args.use_pred_duration and pred_duration is not None)
        gen_num_frames = requested_num_frames if forced_len else None
        caption_num_frames = None
        prefix_frames_arg = None
        # Cross-chunk V2V: at step>=1 splice the previous chunk's last 4k-3
        # frames ahead of this action so the model continues from the previous
        # CHUNK, composed with the context-fill items. Falls back to plain
        # I2V+context when the previous chunk is too short for the requested k.
        if args.cross_chunk_v2v and step_idx > 0 and prev_chunk_frames:
            action_frames = snap_frames_4n1(gen_num_frames if gen_num_frames else args.num_frames)
            timeline = combined_timeline(action_frames, args.cross_chunk_latents)
            reason = prefix_skip_reason(len(prev_chunk_frames), timeline)
            if reason is None:
                p = timeline["prefix_pixel_frames"]
                prefix_frames_arg = prev_chunk_frames[-p:]
                gen_num_frames = timeline["model_frames"]
                caption_num_frames = action_frames
                step_result["cross_chunk_v2v"] = {
                    "prefix_source": "generated_prev_tail",
                    "prefix_latents": args.cross_chunk_latents,
                    "prefix_pixel_frames": p,
                    "action_frames": action_frames,
                    "model_frames": timeline["model_frames"],
                }
            else:
                step_result["cross_chunk_v2v"] = {
                    "prefix_source": "none",
                    "prefix_latents": args.cross_chunk_latents,
                    "skip_reason": reason,
                }
        keep_full_frames = need_ctx_frames or args.cross_chunk_v2v
        # FL2V goal conditioning: when a plan step carries a target frame (the
        # goal image), feed it like the target_frame_condition training source.
        target_frame_arg = None
        if plan.get("target_frame_path"):
            target_frame_arg = _open_rgb_image(plan["target_frame_path"])
            if target_frame_arg is not None and prefix_frames_arg is not None:
                # Both claim a conditioned latent slot of the main video: the prefix the
                # leading k latents, the goal the last one. No training source conditions
                # on both at once, so composing them would be out of distribution. The
                # recipe the caller declared with --cross-chunk-v2v wins, and the drop is
                # recorded instead of being silently invisible.
                step_result["target_frame_suppressed"] = (
                    "cross_chunk_v2v prefix takes precedence over target_frame_path"
                )
                target_frame_arg = None
        last_frame, video_path = generator.generate_visual_plan(
            generator_plan,
            generator_obs,
            step_dir,
            aspect_ratio=initial_ar,
            seed=args.seed,
            write_output=write_outputs,
            context_images=ctx_images,
            keep_frames=keep_full_frames,
            prefix_frames=prefix_frames_arg,
            target_frame=target_frame_arg,
            num_frames=gen_num_frames,
            caption_num_frames=caption_num_frames,
            temporal_caption=gen_temporal_caption,
        )
        generator_time = time.time() - t0
        step_result["video_path"] = str(video_path)
        step_result["generator_time_s"] = round(generator_time, 2)
        current_obs = last_frame
        gen_keyframes.append(last_frame)
        if args.cross_chunk_v2v and getattr(generator, "last_video_frames_pil", None):
            prev_chunk_frames = generator.last_video_frames_pil
        if need_ctx_frames and getattr(generator, "last_video_frames_pil", None):
            ctx_fill_chunks.append(generator.last_video_frames_pil)

        if write_outputs:
            _make_step_comparison_video(
                gt_video_path=plan.get("clip_video_path"),
                pred_video_path=str(video_path),
                gt_text=gt_plan,
                pred_text=predicted_plan,
                step_idx=step_idx,
                output_path=step_dir / "gt_vs_pred.mp4",
            )

        results.append(step_result)
        if args.gt_text_and_length:
            executed_plan = generator_plan
        elif args.use_pred_duration:
            executed_plan = pred_action
        else:
            executed_plan = predicted_plan
        history.append((executed_plan, last_frame))
        log.info(
            f"  [{sample_id}] step {step_idx}/{num_plans-1}: "
            f"pred='{predicted_plan[:80]}...' | gt='{gt_plan[:80]}...' | "
            f"frames={requested_num_frames}->{generated_num_frames}"
        )

        # Write summary after each step (crash-safe)
        if write_outputs:
            with open(summary_path, "w") as f:
                json.dump(_build_episode_summary(episode, results, args), f, indent=2, ensure_ascii=False)

    # The individual step outputs are kept for diagnosis.  Additionally make a
    # presentation-ready continuous episode video.  A successful cross-chunk
    # V2V step begins with the previous clip tail, so the merger reads the
    # per-step prefix metadata and removes that complete overlap before append.
    if write_outputs and results:
        merge_manifest = merge_episode_videos(
            results,
            output_dir / DEFAULT_OUTPUT_NAME,
            fps=args.fps,
        )
        (output_dir / MANIFEST_NAME).write_text(
            json.dumps(merge_manifest, indent=2, ensure_ascii=False) + "\n"
        )
        episode_summary = _build_episode_summary(episode, results, args)
        episode_summary["continuous_video"] = merge_manifest
        with open(summary_path, "w") as f:
            json.dump(episode_summary, f, indent=2, ensure_ascii=False)

    return results


def _open_rgb_image(path: str) -> Image.Image:
    """Load a source observation and release its file descriptor immediately."""
    with Image.open(path) as image:
        return image.convert("RGB")


def _wrap_text(text: str, max_chars: int = 60) -> list[str]:
    """Wrap text into lines of at most max_chars characters."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines


def _make_step_comparison_video(
    gt_video_path: str | None,
    pred_video_path: str | None,
    gt_text: str,
    pred_text: str,
    step_idx: int,
    output_path: Path,
) -> None:
    """Create a side-by-side GT vs Pred comparison video for one step."""
    gt_cap = cv2.VideoCapture(gt_video_path) if gt_video_path and Path(gt_video_path).exists() else None
    pred_cap = cv2.VideoCapture(pred_video_path) if pred_video_path and Path(pred_video_path).exists() else None

    if gt_cap is None and pred_cap is None:
        return

    fps = 24
    src = gt_cap or pred_cap
    fps = src.get(cv2.CAP_PROP_FPS) or 24

    # Read all frames from both videos
    gt_frames = []
    if gt_cap:
        while True:
            ret, frame = gt_cap.read()
            if not ret:
                break
            gt_frames.append(frame)
        gt_cap.release()

    pred_frames = []
    if pred_cap:
        while True:
            ret, frame = pred_cap.read()
            if not ret:
                break
            pred_frames.append(frame)
        pred_cap.release()

    num_frames = max(len(gt_frames), len(pred_frames))
    if num_frames == 0:
        return

    # Use first available frame to determine size
    ref = gt_frames[0] if gt_frames else pred_frames[0]
    frame_h, frame_w = ref.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w * 2, frame_h))

    gt_lines = _wrap_text(f"GT: {gt_text}", max_chars=frame_w // 12)
    pred_lines = _wrap_text(f"Pred: {pred_text}", max_chars=frame_w // 12)

    for i in range(num_frames):
        left = gt_frames[i] if i < len(gt_frames) else np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        right = pred_frames[i] if i < len(pred_frames) else np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        if left.shape[:2] != (frame_h, frame_w):
            left = cv2.resize(left, (frame_w, frame_h))
        if right.shape[:2] != (frame_h, frame_w):
            right = cv2.resize(right, (frame_w, frame_h))

        canvas = np.concatenate([left, right], axis=1)

        # GT text (top-left, green)
        y_offset = 30
        for line in gt_lines:
            cv2.putText(canvas, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 26

        # Pred text (top-right, yellow)
        y_offset = 30
        for line in pred_lines:
            cv2.putText(canvas, line, (frame_w + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (frame_w + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            y_offset += 26

        # Step indicator + divider
        cv2.putText(canvas, f"Step {step_idx}", (frame_w - 40, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (frame_w, 0), (frame_w, frame_h), (255, 255, 255), 2)

        writer.write(canvas)

    writer.release()


def _distributed_context() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _write_rank_results(output_dir: Path, rank: int, results: list[dict]) -> Path:
    partial_dir = output_dir / "partial_results"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partial_dir / f"rank_{rank:04d}.json"
    with open(partial_path, "w") as f:
        json.dump({"rank": rank, "episodes": results}, f, indent=2, ensure_ascii=False)
    return partial_path


def _load_all_rank_results(output_dir: Path, world_size: int) -> list[dict]:
    all_results: list[dict] = []
    partial_dir = output_dir / "partial_results"
    for rank in range(world_size):
        partial_path = partial_dir / f"rank_{rank:04d}.json"
        if not partial_path.exists():
            raise FileNotFoundError(f"Missing data-parallel partial result: {partial_path}")
        payload = json.loads(partial_path.read_text())
        all_results.extend(payload["episodes"])
    return sorted(all_results, key=lambda ep: ep["episode_index"])

# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    rank, world_size = _distributed_context()
    validate_run_args(parser, args)
    validate_data_parallel_args(args, world_size)

    test_data_path = Path(args.test_data)
    episodes = [
        _resolve_record_paths(_normalize_episode(episode), test_data_path.parent)
        for episode in load_test_data(test_data_path)
    ]
    if args.data_seed is not None:
        random.Random(args.data_seed).shuffle(episodes)
        log.info(f"Shuffled episode order with data_seed={args.data_seed}")
    log.info(f"Loaded {len(episodes)} episodes from {args.test_data}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _barrier()

    model = load_model(args)
    from interleave_vp.plan_generator import PlanGeneratorEngine
    from interleave_vp.plan_reasoner import PlanReasonerEngine

    reasoner = PlanReasonerEngine(
        model,
        resolution=args.resolution,
        reasoner_input_size=args.reasoner_input_size,
    )
    generator = PlanGeneratorEngine(
        model,
        resolution=args.resolution,
        num_frames=args.num_frames,
        fps=args.fps,
        num_steps=args.num_steps,
        guidance=args.guidance,
        shift=args.shift,
        anchor_resolution=args.anchor_resolution,
    )

    indexed_episodes = list(enumerate(episodes))
    if args.data_parallel:
        indexed_episodes = indexed_episodes[rank::world_size]
        log.info(
            f"Rank {rank}/{world_size} processing {len(indexed_episodes)} of {len(episodes)} episodes"
        )

    all_results: list[dict] = []
    for local_idx, (ep_idx, episode) in enumerate(indexed_episodes):
        sample_id = episode["id"]
        log.info(
            f"Episode {local_idx + 1}/{len(indexed_episodes)} "
            f"(global {ep_idx + 1}/{len(episodes)}): {sample_id}"
        )
        # Episode ids commonly carry a group separator ("room/bedroom-03"); keep it out of
        # the directory name so one episode is one directory.
        ep_dir = output_dir / f"episode_{ep_idx:04d}_{sample_id.replace('/', '_')}"
        ep_results = evaluate_episode(episode, reasoner, generator, ep_dir, args)
        ep_summary = _build_episode_summary(episode, ep_results, args)
        ep_summary["episode_index"] = ep_idx
        ep_summary["rank"] = rank
        all_results.append(ep_summary)

    if args.data_parallel:
        _write_rank_results(output_dir, rank, all_results)
        _barrier()
        if is_main_rank():
            all_results = _load_all_rank_results(output_dir, world_size)

    if is_main_rank():
        summary = {
            "resolution": args.resolution,
            "data_parallel": args.data_parallel,
            "world_size": world_size,
            "num_input_episodes": len(episodes),
            "num_episodes": len(all_results),
            "total_steps": sum(ep["num_steps_executed"] for ep in all_results),
            "config": {
                **_evaluation_config(args),
                "checkpoint_path": args.checkpoint_path,
                "dp_shard_size": args.dp_shard_size if args.data_parallel else None,
                "dp_replicate_size": (args.dp_replicate_size or world_size) if args.data_parallel else None,
            },
            "episodes": all_results,
        }
        if args.use_pred_duration or args.reasoner_prompt_format == "tags":
            metrics = _duration_metrics(all_results)
            summary["duration_metrics"] = metrics
            log.info(f"Duration metrics: {json.dumps(metrics, ensure_ascii=False)}")
        save_results(summary, output_dir)

    total_steps = sum(ep["num_steps_executed"] for ep in all_results)
    log.info(f"Done: {len(all_results)} episodes, {total_steps} steps")


if __name__ == "__main__":
    main()
