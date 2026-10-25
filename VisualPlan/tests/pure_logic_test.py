# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for the dependency-free layers: frame grid, memory selection, parsing, schema."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interleave_vp.context_fill import build_fill_plan, select_history_units, uniform_pick
from interleave_vp.cross_chunk_v2v import combined_timeline, eligible_prefix_latents, prefix_skip_reason
from interleave_vp.duration_label import duration_label_seconds, frames_from_duration, snap_frames_4n1
from interleave_vp.episodes import (
    build_duration_comparison,
    resolve_record_paths as _resolve,
    build_episode_summary,
    duration_metrics,
    evaluation_config,
    gt_duration_label,
    is_task_completed,
    normalize_episode,
    parse_duration_json,
    resolve_generation_frames,
)
from interleave_vp.text_parsing import (
    DONE_REPLACEMENTS,
    is_done_instruction,
    parse_duration_seconds,
    parse_interleaved_tag_prediction,
)


def test_frame_grid_is_always_4n_plus_1():
    for frames in list(range(1, 260)) + [400, 1000]:
        snapped = snap_frames_4n1(frames)
        assert (snapped - 1) % 4 == 0 and 5 <= snapped <= 241
    assert snap_frames_4n1(38) == 37 and snap_frames_4n1(63) == 61 and snap_frames_4n1(1) == 5


def test_duration_round_trip_losses_exactly_one_second():
    # The label is whole seconds, so the recovered length is within ~12 frames @24fps.
    assert duration_label_seconds(38) == 2 and frames_from_duration(2) == 45
    assert duration_label_seconds(81) == 3 and frames_from_duration(3) == 69
    assert duration_label_seconds(0) == 1
    assert frames_from_duration(0) == 5 and frames_from_duration(999) == 241


def test_cross_chunk_timeline_and_guards():
    assert combined_timeline(81, 4) == {
        "prefix_latents": 4, "prefix_pixel_frames": 13, "action_frames": 81,
        "model_frames": 93, "model_latents": 24,
    }
    assert eligible_prefix_latents(19, 81) == [2, 4]
    assert "needs 29" in prefix_skip_reason(19, combined_timeline(81, 8))
    assert prefix_skip_reason(1000, combined_timeline(237, 2)) is None
    assert "exceeds" in prefix_skip_reason(1000, combined_timeline(241, 8))


def test_context_slots_are_spread_and_end_on_the_most_recent_chunk():
    # Documented cases: [anchor, ...spread..., newest] for K=4 -> the non-anchor units.
    assert select_history_units(4, 4) == [1, 2, 3]
    assert select_history_units(7, 4) == [2, 4, 6]
    assert select_history_units(11, 4) == [3, 7, 10]
    assert select_history_units(44, 4) == [14, 29, 43]
    for n in range(1, 30):
        units = select_history_units(n, 4)
        assert units[-1] == n - 1 and len(set(units)) == len(units)
    assert uniform_pick(0, 3) == [] and select_history_units(0, 4) == []


def test_fill_frames_land_inside_referenced_units():
    plan = build_fill_plan([49, 81], 3)
    assert plan == sorted(plan)
    for unit, idx in plan:
        assert 1 <= idx <= [49, 81][unit] - 1
    assert build_fill_plan([], 2) == []


def test_tag_parsing_degrades_per_field():
    full = parse_interleaved_tag_prediction(
        "<subplan>a</subplan><world_modeling>b</world_modeling><duration>4s</duration>"
    )
    assert full["format"] == "tags" and full["tags_format_ok"] is True
    assert full["subplan"] == "a" and full["duration"] == "4s"

    partial = parse_interleaved_tag_prediction("<subplan>only this</subplan>")
    assert partial["format"] == "partial_tags" and partial["tags_format_ok"] is False
    assert partial["world_modeling"] == "<subplan>only this</subplan>"  # raw fallback keeps text usable
    assert partial["duration"] is None  # no text fallback for a number

    plain = parse_interleaved_tag_prediction("Move to the chair")
    assert plain["format"] == "none" and plain["subplan"] == "Move to the chair"
    assert parse_duration_seconds("about 4.5 seconds") == 4.5
    assert parse_duration_seconds(None) is None


def test_duration_json_is_strict_and_fence_tolerant():
    assert parse_duration_json('{"action": "x", "duration": 3}') == ("x", 3)
    assert parse_duration_json('```json\n{"action": "x", "duration": 3}\n```') == ("x", 3)
    assert parse_duration_json('{"action": "x", "duration": true}') is None  # bool would become 1
    assert parse_duration_json('{"action": "x", "duration": 2.5}') is None  # labels are whole seconds
    assert parse_duration_json("no braces") is None


def test_completion_detection():
    assert is_done_instruction(DONE_REPLACEMENTS[-1])
    assert not is_done_instruction("Open the drawer")
    assert is_task_completed("All work has been done as instructed.")
    assert not is_task_completed("Pick up the pillow")


def test_normalize_episode_sorts_and_rejects_bad_records():
    episode = normalize_episode(
        {"id": "a/b", "high_level_task": "t", "sub_plan": [
            {"plan_index": 1, "instruction": "second", "first_frame_path": "1.jpg"},
            {"plan_index": 0, "instruction": "first", "first_frame_path": "0.jpg"},
        ]}
    )
    assert [p["plan_index"] for p in episode["sub_plan"]] == [0, 1]
    for bad, message in (
        ({"id": "", "high_level_task": "t", "sub_plan": [{}]}, "non-empty string"),
        ({"id": "x", "high_level_task": "t", "sub_plan": []}, "non-empty list"),
        ({"id": "x", "high_level_task": "t", "sub_plan": [
            {"plan_index": 0, "instruction": "a", "first_frame_path": "0.jpg"},
            {"plan_index": 0, "instruction": "b", "first_frame_path": "1.jpg"},
        ]}, "duplicate plan_index"),
        ({"id": "x", "high_level_task": "t", "sub_plan": [{"plan_index": "0", "instruction": "a",
                                                            "first_frame_path": "0.jpg"}]}, "must be an integer"),
    ):
        try:
            normalize_episode(bad)
        except ValueError as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError(f"expected rejection containing {message!r}")


def test_resolve_generation_frames_teacher_forcing():
    assert resolve_generation_frames({"frame_count": 38}, 81, False) == (81, 81)
    assert resolve_generation_frames({"frame_count": 38}, 81, True) == (38, 37)
    assert resolve_generation_frames({"frame_count": 400}, 81, True) == (241, 241)
    try:
        resolve_generation_frames({}, 81, True)
    except ValueError as exc:
        assert "frame_count" in str(exc)
    else:
        raise AssertionError("teacher forcing without frame_count must fail")


def test_gt_duration_labels_and_metrics():
    assert gt_duration_label({"instruction": "x", "frame_count": 81, "fps": 24}) == 3
    assert gt_duration_label({"instruction": DONE_REPLACEMENTS[0]}) == 0  # terminal step
    assert gt_duration_label({"instruction": "x"}) is None  # unlabelled -> metric skipped
    comparison = build_duration_comparison(
        {"instruction": "x", "frame_count": 81, "fps": 24}, '{"action": "x", "duration": 3}', ("x", 3)
    )
    assert comparison["gt"]["duration_s"] == 3 and comparison["comparison"]["duration_exact"] is True
    steps = [
        {"duration_parse_ok": True, "pred_duration_s": 3, "gt_duration_s": 3},
        {"duration_parse_ok": False, "pred_duration_s": None, "gt_duration_s": 2},
    ]
    metrics = duration_metrics([{"steps": steps}])
    assert metrics["steps"] == 2 and metrics["parse_failures"] == 1
    assert metrics["compared"] == 1 and metrics["exact_match"] == 1.0
    assert metrics["gt_duration_hist"] == {"2": 1, "3": 1}


def test_evaluation_config_records_the_recipe():
    class Args:
        resolution, max_plan_steps, history_window, data_seed = "480", 15, 3, 8
        max_new_tokens, do_sample, temperature, top_p = 512, False, 0.2, 0.9
        num_frames, fps, num_steps, guidance, shift, seed = 81, 24, 50, 5.0, 5.0, 42
        context_frames, context_fill, anchor_resolution = 4, True, "720"
        use_gt_plan, gt_text_and_length, use_pred_duration = False, False, True
        pred_duration_offset, reasoner_prompt_format = 0.5, "tags"
        reasoner_input_size, generator_world_modeling = None, True
        cross_chunk_v2v, cross_chunk_latents = True, 4

    config = evaluation_config(Args())
    assert config["context_fill"] is True and config["anchor_resolution"] == "720"
    assert config["cross_chunk_v2v"] is True and config["cross_chunk_latents"] == 4
    assert config["pred_duration_offset_s"] == 0.5 and config["reasoner_input_size"] is None
    # Dropped knobs must not be recorded any more.
    for banned in ("history_chunks", "kv_cache", "pinned_sinks", "memory_mode", "gt_frame_offset",
                   "gt_current_observation"):
        assert banned not in config, banned
    summary = build_episode_summary({"id": "e", "high_level_task": "t", "sub_plan": [
        {"instruction": "a", "plan_index": 0, "first_frame_path": "0.jpg"}]}, [], Args())
    assert summary["num_steps_executed"] == 0 and summary["config"] is not None


def test_relative_record_paths_resolve_against_the_jsonl_directory():
    from pathlib import Path as P

    episode = {"id": "x", "high_level_task": "t", "sub_plan": [
        {"plan_index": 0, "instruction": "a", "first_frame_path": "assets/p0.jpg",
         "clip_video_path": "assets/p0.mp4", "target_frame_path": "/already/abs.jpg"}]}
    resolved = _resolve(episode, P("/data/bench"))
    step = resolved["sub_plan"][0]
    assert step["first_frame_path"] == "/data/bench/assets/p0.jpg"
    assert step["clip_video_path"] == "/data/bench/assets/p0.mp4"
    assert step["target_frame_path"] == "/already/abs.jpg"  # absolute stays absolute
    assert episode["sub_plan"][0]["first_frame_path"] == "assets/p0.jpg"  # input untouched
