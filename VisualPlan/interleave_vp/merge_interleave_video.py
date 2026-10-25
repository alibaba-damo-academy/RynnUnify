#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Merge interleaved rollout chunks into one continuous episode video.

Cross-chunk V2V clips are not independent videos: a successful transition
starts its output with ``prefix_pixel_frames`` copied from the previous
generated chunk.  This utility removes that complete prefix before appending
the new clip, so the resulting video has no repeated V2V tail frames.

It can be used directly after a rollout, for example::

    python -m interleave_vp.merge_interleave_video \
        --input-dir /path/to/eval_run

The interleaved evaluator also calls :func:`merge_episode_videos`
automatically at the end of every episode.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2


DEFAULT_OUTPUT_NAME = "continuous_visual_plan.mp4"
MANIFEST_NAME = "continuous_visual_plan.json"


def _cross_v2v_prefix_frames(step: Mapping[str, Any]) -> int:
    """Return the known repeated V2V prefix length for one generated step.

    A skipped V2V transition records ``prefix_source='none'`` and contributes
    zero.  Normal I2V chunks also contribute zero: their first decoded frame
    may visually resemble the previous final frame, but is not an explicit
    cross-chunk prefix and must not be discarded blindly.
    """
    transition = step.get("cross_chunk_v2v")
    if not isinstance(transition, Mapping):
        return 0
    if transition.get("prefix_source") != "generated_prev_tail":
        return 0
    prefix_frames = transition.get("prefix_pixel_frames")
    if isinstance(prefix_frames, bool) or not isinstance(prefix_frames, int) or prefix_frames <= 0:
        raise ValueError(
            "successful cross_chunk_v2v metadata must contain a positive "
            f"prefix_pixel_frames, got {prefix_frames!r}"
        )
    return prefix_frames


def _ordered_video_steps(steps: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Keep generated steps, ordered and validated by their rollout index."""
    generated = [step for step in steps if isinstance(step.get("video_path"), str) and step["video_path"]]
    if not generated:
        return []

    try:
        ordered = sorted(generated, key=lambda step: int(step["step"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("every generated step must contain an integer 'step'") from exc
    indices = [int(step["step"]) for step in ordered]
    if len(indices) != len(set(indices)):
        raise ValueError(f"duplicate generated step indices: {indices}")
    return ordered


def merge_episode_videos(
    steps: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    fps: float | None = None,
) -> dict[str, Any]:
    """Merge one episode's ``visual_plan.mp4`` files into a continuous MP4.

    Args:
        steps: Records from an episode's ``summary.json`` (or the evaluator's
            in-memory results).  Each generated record needs ``step`` and
            ``video_path``.  Successful cross-V2V records additionally carry
            ``cross_chunk_v2v.prefix_pixel_frames``.
        output_path: Destination MP4 path.
        fps: Output frame rate.  When omitted, use the first input video's FPS.

    Returns:
        A JSON-serialisable manifest.  For a successful cross-V2V transition
        all prefix frames are removed, including the shared boundary frame:
        appending only ``video[prefix_pixel_frames:]`` avoids both the full
        prefix repetition and a duplicate boundary frame.
    """
    ordered_steps = _ordered_video_steps(steps)
    if not ordered_steps:
        raise ValueError("no generated video_path entries were supplied")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.partial{destination.suffix}")
    temporary.unlink(missing_ok=True)

    writer = None
    output_size: tuple[int, int] | None = None
    output_fps = float(fps) if fps is not None else None
    source_frames = 0
    written_frames = 0
    skipped_v2v_prefix_frames = 0
    chunks: list[dict[str, Any]] = []

    try:
        for step in ordered_steps:
            video_path = Path(str(step["video_path"]))
            if not video_path.is_file():
                raise FileNotFoundError(f"step {step['step']} video does not exist: {video_path}")

            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"cannot decode step {step['step']} video: {video_path}")
            try:
                input_fps = float(capture.get(cv2.CAP_PROP_FPS))
                if input_fps <= 0:
                    raise RuntimeError(f"step {step['step']} has invalid FPS: {input_fps}")
                if output_fps is None:
                    output_fps = input_fps
                elif abs(input_fps - output_fps) > 0.01:
                    raise ValueError(
                        f"step {step['step']} FPS is {input_fps}, expected {output_fps}; "
                        "cannot concatenate clips with different frame rates"
                    )

                drop_prefix = _cross_v2v_prefix_frames(step)
                input_count = 0
                appended_count = 0
                while True:
                    ok, frame_bgr = capture.read()
                    if not ok:
                        break
                    if output_size is None:
                        output_size = (frame_bgr.shape[1], frame_bgr.shape[0])
                        writer = cv2.VideoWriter(
                            str(temporary),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            output_fps,
                            output_size,
                        )
                        if not writer.isOpened():
                            writer.release()
                            writer = None
                            raise RuntimeError(
                                "OpenCV could not create the MP4 writer; "
                                "check that the runtime has MP4/FFmpeg codec support"
                            )
                    elif (frame_bgr.shape[1], frame_bgr.shape[0]) != output_size:
                        raise ValueError(
                            f"step {step['step']} size is {frame_bgr.shape[1]}x{frame_bgr.shape[0]}, "
                            f"expected {output_size[0]}x{output_size[1]}"
                        )

                    if input_count >= drop_prefix:
                        # Keep OpenCV's native BGR format for VideoWriter.
                        writer.write(frame_bgr)
                        appended_count += 1
                    input_count += 1

                if input_count == 0:
                    raise RuntimeError(f"step {step['step']} video contains no frames: {video_path}")
                if drop_prefix >= input_count:
                    raise ValueError(
                        f"step {step['step']} has {input_count} frames, but V2V prefix trim requests "
                        f"{drop_prefix}; no continuation would remain"
                    )
                source_frames += input_count
                skipped_v2v_prefix_frames += drop_prefix
                written_frames += appended_count
                chunks.append(
                    {
                        "step": int(step["step"]),
                        "video_path": str(video_path),
                        "video_path_resolution": step.get("_video_path_resolution", "summary"),
                        "input_frames": input_count,
                        "dropped_cross_v2v_prefix_frames": drop_prefix,
                        "appended_frames": appended_count,
                    }
                )
            finally:
                capture.release()

        if writer is None or output_size is None or written_frames == 0:
            raise RuntimeError("no frames were written to the merged video")
        writer.release()
        writer = None
        temporary.replace(destination)
    except Exception:
        if writer is not None:
            writer.release()
        temporary.unlink(missing_ok=True)
        raise

    return {
        "output_path": str(destination),
        "fps": output_fps,
        "width": output_size[0],
        "height": output_size[1],
        "input_chunks": len(chunks),
        "input_frames": source_frames,
        "dropped_cross_v2v_prefix_frames": skipped_v2v_prefix_frames,
        "output_frames": written_frames,
        "chunks": chunks,
    }


def _resolve_episode_video_paths(
    episode_path: Path,
    steps: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Repair stale absolute summary paths from a copied or renamed eval run.

    Evaluator summaries normally contain absolute ``video_path`` values.  If a
    complete run directory is copied/renamed, those paths still point to the
    original run.  Prefer the recorded path when it exists; only when it is
    absent, recover the conventional sibling output
    ``episode_dir/step_NNN/visual_plan.mp4``.
    """
    resolved_steps: list[dict[str, Any]] = []
    for raw_step in steps:
        if not isinstance(raw_step, Mapping):
            raise ValueError("every summary step must be an object")
        step = dict(raw_step)
        video_value = step.get("video_path")
        if not isinstance(video_value, str) or not video_value:
            resolved_steps.append(step)
            continue
        recorded_path = Path(video_value)
        if recorded_path.is_file():
            resolved_steps.append(step)
            continue
        try:
            step_index = int(step["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("a generated summary step must contain an integer 'step'") from exc
        local_path = episode_path / f"step_{step_index:03d}" / "visual_plan.mp4"
        if local_path.is_file():
            step["video_path"] = str(local_path)
            step["_video_path_resolution"] = "episode_local_fallback"
            step["_summary_video_path"] = str(recorded_path)
        resolved_steps.append(step)
    return resolved_steps


def merge_episode_directory(
    episode_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load an evaluator episode directory and merge its generated chunks."""
    episode_path = Path(episode_dir)
    summary_path = episode_path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"episode summary does not exist: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, Mapping) or not isinstance(summary.get("steps"), list):
        raise ValueError(f"invalid episode summary (missing list 'steps'): {summary_path}")
    config = summary.get("config")
    summary_fps = config.get("fps") if isinstance(config, Mapping) else None
    if isinstance(summary_fps, bool) or not isinstance(summary_fps, (int, float)):
        summary_fps = None
    destination = Path(output_path) if output_path is not None else episode_path / DEFAULT_OUTPUT_NAME
    resolved_steps = _resolve_episode_video_paths(episode_path, summary["steps"])
    manifest = merge_episode_videos(resolved_steps, destination, fps=summary_fps)
    (episode_path / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def _merge_episode_worker(episode_dir: str) -> dict[str, Any]:
    """Process-pool entry point; keep it top-level so it is pickle-safe."""
    try:
        manifest = merge_episode_directory(episode_dir)
        return {"episode_dir": episode_dir, "status": "success", "manifest": manifest}
    except Exception as exc:  # report every broken episode in a batch run
        return {"episode_dir": episode_dir, "status": "error", "error": str(exc)}


def merge_run_directory(input_dir: str | Path, *, workers: int = 8) -> dict[str, Any]:
    """Merge every evaluator episode directly below one test-output directory.

    Each successful result is written inside its own ``episode_*`` folder, next
    to that folder's ``summary.json``.  Processing continues after an invalid
    episode so one damaged output does not prevent all other episodes from
    receiving their continuous videos.  With ``workers > 1``, independent
    episode directories are decoded and written in separate processes.
    """
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    run_path = Path(input_dir)
    if not run_path.is_dir():
        raise NotADirectoryError(f"test output directory does not exist: {run_path}")
    episode_dirs = sorted(
        path for path in run_path.iterdir()
        if path.is_dir() and path.name.startswith("episode_") and (path / "summary.json").is_file()
    )
    if not episode_dirs:
        raise ValueError(f"no episode_*/summary.json directories found in: {run_path}")

    episode_paths = [str(episode_dir) for episode_dir in episode_dirs]
    total = len(episode_paths)
    print(f"[merge] found {total} episodes; workers={workers}", flush=True)
    if workers == 1:
        episodes = []
        for completed, episode_dir in enumerate(episode_paths, start=1):
            entry = _merge_episode_worker(episode_dir)
            episodes.append(entry)
            print(
                f"[merge] {completed}/{total} {Path(episode_dir).name}: {entry['status']}",
                flush=True,
            )
    else:
        # Keep results sorted by episode path, while completed futures provide
        # real-time progress even if an early episode is slower than the rest.
        episodes = [{} for _ in episode_paths]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_merge_episode_worker, episode_dir): (index, episode_dir)
                for index, episode_dir in enumerate(episode_paths)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                index, episode_dir = futures[future]
                entry = future.result()
                episodes[index] = entry
                print(
                    f"[merge] {completed}/{total} {Path(episode_dir).name}: {entry['status']}",
                    flush=True,
                )
    return {
        "input_dir": str(run_path),
        "workers": workers,
        "episodes_found": len(episode_dirs),
        "episodes_merged": sum(entry["status"] == "success" for entry in episodes),
        "episodes_failed": sum(entry["status"] == "error" for entry in episodes),
        "episodes": episodes,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge interleaved step videos into one continuous MP4")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--episode-dir",
        help="One episode directory containing summary.json and step_* outputs",
    )
    input_group.add_argument(
        "--input-dir",
        help="One test-output directory; merge every direct episode_* child containing summary.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Destination MP4 for --episode-dir only (default: <episode-dir>/{DEFAULT_OUTPUT_NAME})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel episode merge processes for --input-dir only (default: 8)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.input_dir is not None:
        if args.output is not None:
            raise SystemExit("--output can only be used with --episode-dir")
        workers = 8 if args.workers is None else args.workers
        manifest = merge_run_directory(args.input_dir, workers=workers)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        if manifest["episodes_failed"]:
            raise SystemExit(1)
    else:
        if args.workers not in (None, 1):
            raise SystemExit("--workers is only valid with --input-dir")
        manifest = merge_episode_directory(args.episode_dir, args.output)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
