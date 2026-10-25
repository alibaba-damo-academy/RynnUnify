#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prompt constants, caption JSON, image preprocessing, rank helpers.

Both engines share these helpers. Prompt templates and caption serialization
must stay aligned with the checkpoint's training format.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from cosmos_framework.utils import log


# ── Distributed helpers ────────────────────────────────────────────────────

def is_main_rank() -> bool:
    """True on rank 0 or when not running under torchrun."""
    return not dist.is_initialized() or dist.get_rank() == 0


# ── Training-aligned prompt constants ─────────────

TASK_PROMPT_TEMPLATE = (
    "You are an embodied planning agent. "
    "Given the high-level task, the initial state, and the execution trajectory of completed action-result pairs, predict the next executable action. "
    "Use the most recent visual observation as the current state. "
    "Output only the action instruction."
)

# Used with --use-pred-duration; keep this prompt identical to the training format.
TASK_PROMPT_TEMPLATE_DURATION = (
    "You are an embodied planning agent. "
    "Given the high-level task, the initial state, and the execution trajectory of completed action-result pairs, predict the next executable action and estimate its duration in whole seconds. "
    "Use the most recent visual observation as the current state. "
    "Output only a JSON object: {\"action\": \"<action instruction>\", \"duration\": <integer seconds>}."
)

DEFAULT_HISTORY_WINDOW = 3


# ── Shared helpers ──────────────────────────────────────────────────────────


def resize_and_center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize and center crop at PIL level to match Reasoner training preprocessing."""
    input_w, input_h = img.size
    resize_ratio = max(target_w / input_w, target_h / input_h)
    resize_w = round(input_w * resize_ratio)
    resize_h = round(input_h * resize_ratio)
    img = img.resize((resize_w, resize_h), Image.LANCZOS)
    crop_x = round((resize_w - target_w) / 2)
    crop_y = round((resize_h - target_h) / 2)
    return img.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))


def build_caption_json(
    plan: str,
    target_w: int,
    target_h: int,
    ar_bucket: str,
    duration_sec: float,
    fps: int,
    temporal_caption: str | None = None,
) -> dict:
    """Build the Generator's training-aligned caption JSON.

    ``temporal_caption`` optionally describes future visual evolution. Preserve
    the training key order: actions, temporal caption, then media metadata.
    """
    caption: dict = {"actions": [{"description": plan}]}
    if temporal_caption:
        caption["temporal_caption"] = temporal_caption
    caption.update(
        {
            "resolution": {"H": target_h, "W": target_w},
            "aspect_ratio": ar_bucket,
            "duration": f"{duration_sec:.1f}s",
            "fps": fps,
        }
    )
    return caption


def extract_last_frame(video_cthw: torch.Tensor) -> Image.Image:
    """Extract the last frame from a [C,T,H,W] float tensor in [0,1] → PIL."""
    last = video_cthw[:, -1, :, :]  # [C,H,W]
    arr = (last.clamp(0, 1).cpu().float().numpy() * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # [H,W,C]
    return Image.fromarray(arr, mode="RGB")


# ── Data I/O ────────────────────────────────────────────────────────────────


def load_test_data(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else [data]


def save_results(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"Results saved to {output_dir / 'summary.json'}")
