# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CLI surface and argument validation — torch-free on purpose.

Keeping the parser and its guards out of the GPU entry point means the exact flag
set, its defaults, and every "these two knobs cannot coexist" rule are verifiable
on a machine with no CUDA and no checkpoint. It is also what lets ``tests/``
cross-check the launcher script against the parser.
"""

from __future__ import annotations

import argparse

# Resolution tiers the pipeline ships. Must stay a subset of the framework's
# VIDEO_RES_SIZE_INFO; the engines use the framework table for actual sizes.
RESOLUTION_TIERS = ("256", "480", "704", "720")

# Trained inference settings. Launcher agreement is checked by tests/cli_contract_test.py.
V3_RECIPE = {
    "resolution": "480",
    "shift": 5.0,
    "num_steps": 50,
    "guidance": 5.0,
    "fps": 24,
    "max_plan_steps": 15,
    "context_frames": 4,
    "context_fill": True,
    "anchor_resolution": "720",
    "cross_chunk_v2v": True,
    "cross_chunk_latents": 4,
    "use_pred_duration": True,
    "reasoner_prompt_format": "tags",
    "pred_duration_offset": 0.5,
    "generator_world_modeling": True,
}

DEFAULT_HISTORY_WINDOW = 3


def parse_reasoner_input_size(value: str) -> tuple[int, int]:
    """Parse the optional ``WIDTHxHEIGHT`` resize applied before Reasoner encoding."""
    normalized = str(value).strip().lower()
    try:
        width_text, height_text = normalized.split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"expected WIDTHxHEIGHT, got {value!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"reasoner input dimensions must be positive, got {value!r}")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interleaved Reasoner/Generator rollout with scene memory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data / output")
    data.add_argument("--test-data", required=True, help="JSONL or JSON of episodes (schema in README)")
    data.add_argument("--output-dir", required=True, help="Output directory")
    data.add_argument("--data-seed", type=int, default=None, help="Shuffle episode order with this seed")
    data.add_argument(
        "--max-plan-steps",
        type=int,
        default=None,
        help="Cap on evaluated steps per episode (default: every labelled step in the record)",
    )

    model = parser.add_argument_group("checkpoint")
    model.add_argument(
        "--checkpoint-path",
        required=True,
        help="Omni checkpoint: HF export (config.json + safetensors) or DCP directory with --experiment",
    )
    model.add_argument("--config-file", default=None, help="Model config YAML (auto-detected for HF checkpoints)")
    model.add_argument(
        "--experiment",
        default="",
        help="Hydra experiment name for DCP checkpoints (e.g. plan_generator_sft_nano). Uses the training "
        "MODULE config so the architecture matches the checkpoint exactly; with --experiment set and "
        "--config-file absent, config_file defaults to cosmos_framework/configs/base/config.py.",
    )
    model.add_argument(
        "--experiment-override",
        action="append",
        default=[],
        help="Hydra override on top of --experiment (repeatable), e.g. "
        "--experiment-override model.config.tokenizer.vae_path=/abs/Wan2.2_VAE.pth",
    )
    model.add_argument(
        "--vlm-processor-path",
        default=None,
        help="HF directory holding the multimodal Qwen3-VL processor. Needed when the checkpoint exposes a "
        "text-only tokenizer (also reads INTERLEAVE_VLM_PROCESSOR_PATH).",
    )

    video = parser.add_argument_group("generator sampling")
    video.add_argument("--resolution", default=V3_RECIPE["resolution"], choices=list(RESOLUTION_TIERS))
    video.add_argument("--num-frames", type=int, default=81, help="Default chunk length; snapped down to 4n+1")
    video.add_argument("--fps", type=int, default=V3_RECIPE["fps"])
    video.add_argument("--num-steps", type=int, default=V3_RECIPE["num_steps"], help="UniPC sampling steps")
    video.add_argument("--guidance", type=float, default=V3_RECIPE["guidance"], help="Classifier-free guidance scale")
    video.add_argument("--shift", type=float, default=V3_RECIPE["shift"], help="Noise schedule shift (tier-dependent)")
    video.add_argument("--seed", type=int, default=42)

    memory = parser.add_argument_group("scene memory (must match the trained recipe)")
    memory.add_argument(
        "--context-frames",
        type=int,
        default=0,
        help="K fully-conditioned context items: episode anchor + generated keyframes packed before the "
        "main video. Only with a --context-frames-trained checkpoint (0 = off)",
    )
    memory.add_argument(
        "--context-fill",
        action="store_true",
        help="Choose the context slots as [anchor] + [chunks spread evenly "
        "over the generated history] + [most recent chunk], topping up early steps with interior frames so "
        "every step feeds exactly K items. Must match the checkpoint's training context layout.",
    )
    memory.add_argument(
        "--anchor-resolution",
        default="",
        choices=[""] + list(RESOLUTION_TIERS),
        help="Tier for context item 0 (the episode anchor), e.g. 720 while the main video stays at "
        "--resolution. MUST match the training-side history_anchor_resolution. Empty = same tier as "
        "the context items (256).",
    )

    continuity = parser.add_argument_group("cross-chunk continuity")
    continuity.add_argument(
        "--cross-chunk-v2v",
        action="store_true",
        help="At step>=1 hand the previous generated chunk's last 4k-3 frames over clean as a V2V prefix "
        "ahead of the current action, composed with the context items. Falls back to plain I2V+context when "
        "the previous chunk is too short (recorded as skip_reason, never padded).",
    )
    continuity.add_argument(
        "--cross-chunk-latents",
        type=int,
        default=V3_RECIPE["cross_chunk_latents"],
        choices=[2, 4, 8],
        help="Clean prefix length in LATENT frames k (4k-3 pixels)",
    )

    reasoner = parser.add_argument_group("reasoner / text conditioning")
    reasoner.add_argument("--history-window", type=int, default=DEFAULT_HISTORY_WINDOW,
                          help="Action-result pairs kept in the Reasoner prompt")
    reasoner.add_argument(
        "--reasoner-prompt-format",
        choices=("bare", "tags"),
        default="bare",
        help="'bare': [EXECUTION TRAJECTORY] with interleaved result images, action-only answer. "
        "'tags': the training-native three-tag conversation (text-only [PREVIOUS STEPS] + one "
        "[CURRENT OBSERVATION] frame), answered as <subplan>/<world_modeling>/<duration>.",
    )
    reasoner.add_argument(
        "--generator-world-modeling",
        action="store_true",
        help="With --reasoner-prompt-format tags: put the predicted <world_modeling> into the caption_json "
        "as 'temporal_caption', the dense-description field the training plan sources carry.",
    )
    reasoner.add_argument(
        "--use-pred-duration",
        action="store_true",
        help="Let the predicted duration drive the chunk length "
        "(frames = round(d*fps) floored to 4n+1, clamped [5,241]); parse failures fall back to "
        "--num-frames and are counted. Mutually exclusive with --gt-text-and-length.",
    )
    reasoner.add_argument(
        "--pred-duration-offset",
        type=float,
        default=0.0,
        help="Seconds added to a parsed tags <duration> before conversion to frames (tags + "
        "--use-pred-duration only). Recorded metrics stay unadjusted, so this tuning knob cannot "
        "inflate the duration accuracy.",
    )
    reasoner.add_argument(
        "--gt-text-and-length",
        action="store_true",
        help="Teacher-forcing upper bound: feed each clip's GT instruction and GT frame_count to the "
        "Generator. Predictions are still recorded but cannot stop the rollout early.",
    )
    reasoner.add_argument(
        "--use-gt-plan",
        action="store_true",
        help="Feed GT plan text to the Generator while keeping the predicted chunk length.",
    )
    reasoner.add_argument(
        "--reasoner-input-size",
        type=parse_reasoner_input_size,
        default=None,
        metavar="WIDTHxHEIGHT",
        help="Resize+center-crop every image handed to the Reasoner before its processor, e.g. 1920x1080 "
        "(default: the --resolution tier's bucket geometry). Generator inputs are unchanged.",
    )
    reasoner.add_argument("--max-new-tokens", type=int, default=512)
    reasoner.add_argument("--do-sample", action="store_true")
    reasoner.add_argument("--temperature", type=float, default=1.0)
    reasoner.add_argument("--top-p", type=float, default=None, help="Optional nucleus sampling for the Reasoner")

    parallel = parser.add_argument_group("parallelism")
    parallel.add_argument(
        "--data-parallel",
        action="store_true",
        help="Shard episodes across torchrun ranks; each rank loads a full model replica",
    )
    parallel.add_argument("--dp-shard-size", type=int, default=1)
    parallel.add_argument("--dp-replicate-size", type=int, default=0, help="0 means WORLD_SIZE")
    parallel.add_argument("--cp-size", type=int, default=1)
    parallel.add_argument("--cfgp-size", type=int, default=1)
    return parser


def validate_run_args(parser: argparse.ArgumentParser, args) -> None:
    """Refuse knob combinations that cannot describe the checkpoint being loaded.

    A memory knob that does not match training does not raise inside the model: it
    silently feeds an unseen input distribution, which reads back as a quality
    regression. Refusing at startup is the only way to catch it.
    """
    if args.history_window <= 0:
        parser.error("--history-window must be a positive integer")
    if args.num_frames < 5:
        parser.error(f"--num-frames must be at least 5 (the VAE needs >= 2 latents), got {args.num_frames}")
    if args.num_frames > 241:
        parser.error(f"--num-frames must be at most 241, got {args.num_frames}")
    if args.context_fill and args.context_frames <= 0:
        parser.error("--context-fill requires --context-frames > 0")
    if args.use_pred_duration and args.gt_text_and_length:
        parser.error("--use-pred-duration and --gt-text-and-length are mutually exclusive")
    if args.pred_duration_offset and (not args.use_pred_duration or args.reasoner_prompt_format != "tags"):
        parser.error("--pred-duration-offset requires --use-pred-duration and --reasoner-prompt-format tags")
    if args.generator_world_modeling and args.reasoner_prompt_format != "tags":
        parser.error("--generator-world-modeling requires --reasoner-prompt-format tags")
    if args.cross_chunk_v2v:
        if not (args.context_frames > 0 and args.context_fill):
            parser.error("--cross-chunk-v2v requires --context-frames and --context-fill")


def validate_data_parallel_args(args, world_size: int) -> None:
    if not args.data_parallel:
        return
    invalid = {
        "--dp-shard-size": args.dp_shard_size,
        "--cp-size": args.cp_size,
        "--cfgp-size": args.cfgp_size,
    }
    invalid = {name: value for name, value in invalid.items() if value != 1}
    if invalid:
        joined = ", ".join(f"{name}={value}" for name, value in invalid.items())
        raise ValueError(
            "--data-parallel shards episodes by rank, so model-parallel axes must stay at 1; got "
            f"{joined}. Set DATA_PARALLEL=0 for model-parallel inference."
        )
    if args.dp_replicate_size not in (0, world_size):
        raise ValueError(
            f"--dp-replicate-size must be 0 or WORLD_SIZE when --data-parallel is enabled; got "
            f"{args.dp_replicate_size}, WORLD_SIZE={world_size}."
        )
