# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CLI defaults, validation, and launcher/parser agreement.

All of this runs without torch, CUDA or a checkpoint — which is exactly why
:mod:`interleave_vp.cli_args` is kept free of those imports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from interleave_vp.cli_args import RESOLUTION_TIERS, V3_RECIPE, build_parser, parse_reasoner_input_size, validate_data_parallel_args, validate_run_args

BASE = ["--test-data", "e.jsonl", "--output-dir", "out", "--checkpoint-path", "ckpt"]


def parse(extra=()):
    return build_parser().parse_args(BASE + list(extra))


def check_rejected(extra, message):
    """``extra`` must be refused, and the complaint must name the offending knob.

    ``parser.error`` writes to stderr and exits; plain ``ValueError`` carries the text.
    Both paths are captured so a silent acceptance is impossible.
    """
    import contextlib
    import io

    needle = message.split()[0]
    parser = build_parser()
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            args = parser.parse_args(BASE + list(extra))
        except SystemExit:
            assert needle in stderr.getvalue(), stderr.getvalue()
            return
        try:
            validate_run_args(parser, args)
        except (SystemExit, ValueError) as exc:
            assert needle in stderr.getvalue() + str(exc), (message, stderr.getvalue() + str(exc))
            return
    raise AssertionError(f"expected rejection mentioning {message!r} for {extra}")


def test_v3_defaults_match_the_training_recipe():
    args = parse()
    assert args.resolution == V3_RECIPE["resolution"] == "480"
    assert (args.num_steps, args.guidance, args.shift, args.fps) == (50, 5.0, 5.0, 24)
    assert args.cross_chunk_latents == V3_RECIPE["cross_chunk_latents"] == 4
    assert args.history_window == 3 and args.max_new_tokens == 512
    assert args.max_plan_steps is None  # no hidden cap when the caller says nothing


def test_v3_command_line_validates():
    args = parse(["--context-frames", "4", "--context-fill", "--anchor-resolution", "720",
                  "--cross-chunk-v2v", "--use-pred-duration", "--reasoner-prompt-format", "tags",
                  "--pred-duration-offset", "0.5", "--generator-world-modeling", "--max-plan-steps", "15"])
    validate_run_args(parser=build_parser(), args=args)
    assert args.context_fill and args.cross_chunk_v2v and args.generator_world_modeling


def test_memory_knobs_that_cannot_match_a_recipe_are_refused():
    check_rejected(["--context-fill"], "--context-fill requires --context-frames")
    check_rejected(["--cross-chunk-v2v"], "--cross-chunk-v2v requires --context-frames")
    check_rejected(["--cross-chunk-v2v", "--context-frames", "4"], "--context-fill")


def test_text_conditioning_switches_are_consistent():
    check_rejected(["--use-pred-duration", "--gt-text-and-length"], "mutually exclusive")
    check_rejected(["--pred-duration-offset", "0.5"], "--pred-duration-offset requires")
    check_rejected(["--use-pred-duration", "--pred-duration-offset", "0.5"], "tags")
    check_rejected(["--generator-world-modeling"], "--generator-world-modeling requires")


def test_frame_and_window_bounds():
    check_rejected(["--num-frames", "3"], "at least 5")
    check_rejected(["--num-frames", "242"], "at most 241")
    check_rejected(["--history-window", "0"], "positive integer")


def test_dropped_knobs_no_longer_exist():
    parser = build_parser()
    known = {option for action in parser._actions for option in action.option_strings}
    for banned in ("--memory-mode", "--memory-count", "--memory-max-frames", "--history-chunks",
                   "--kv-cache", "--pinned-sinks", "--gt-frame-offset", "--gt-current-observation",
                   "--evaluation-mode", "--reasoner-checkpoint-path"):
        assert banned not in known, banned
    try:
        parser.parse_args(BASE + ["--kv-cache"])
    except SystemExit:
        pass
    else:
        raise AssertionError("--kv-cache should be an unknown flag")


def test_resolution_tiers_are_declared():
    assert set(RESOLUTION_TIERS) >= {"256", "480", "720"}
    args = parse(["--anchor-resolution", "720"])
    assert args.reasoner_input_size is None
    assert parse_reasoner_input_size("1920x1080") == (1920, 1080)
    for bad in ("0x100", "wide", "1920"):
        try:
            parse_reasoner_input_size(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_data_parallel_axes_must_stay_one():
    args = parse(["--data-parallel"])
    validate_data_parallel_args(args, world_size=8)  # defaults are legal
    args.dp_shard_size = 2
    try:
        validate_data_parallel_args(args, 8)
    except ValueError as exc:
        assert "must stay at 1" in str(exc)
    else:
        raise AssertionError("dp_shard_size=2 with --data-parallel must fail")
    args = parse(["--data-parallel"])
    args.dp_shard_size = 1
    try:
        validate_data_parallel_args(args, 4)
    except ValueError as exc:
        assert "WORLD_SIZE" in str(exc)


def test_argparse_errors_are_not_swallowed():
    parser = build_parser()
    try:
        parser.parse_args(["--test-data", "x"])  # missing --output-dir / --checkpoint-path
    except SystemExit:
        return
    raise AssertionError("required arguments must be enforced")
