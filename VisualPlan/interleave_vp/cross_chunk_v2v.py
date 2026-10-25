# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cross-chunk V2V frame arithmetic for continuous video generation.

A cross-chunk sample continues the CURRENT clip while conditioning on the previous
chunk's final ``4k-3`` pixel frames (k clean latents of the causal Wan2.2 VAE). The
model timeline is ``[previous tail] ++ [current continuation]`` with the boundary
frame shared, so it is a single contiguous video encoded in one VAE pass.

The inference timeline and conditioning layout must match checkpoint training.
"""

from __future__ import annotations

from interleave_vp.duration_label import MAX_FRAMES

_TEMPORAL_COMPRESSION = 4
ALLOWED_PREFIX_LATENTS = (2, 4, 8)


def pixel_frames_for_latents(k: int) -> int:
    """Pixel frames spanned by the first k clean latents of a causal VAE (= 4k-3)."""
    return _TEMPORAL_COMPRESSION * (k - 1) + 1


def combined_timeline(action_frames: int, prefix_latents: int) -> dict[str, int]:
    """Timeline for ``[previous tail] ++ [current continuation]``.

    ``action_frames`` is the current clip's own length (already snapped to 4n+1).
    The previous tail contributes its k clean latents; the boundary frame is shared,
    so ``model_frames = prefix_pixel_frames + action_frames - 1`` is itself 4n+1.
    """
    prefix_pixel_frames = pixel_frames_for_latents(prefix_latents)
    model_frames = prefix_pixel_frames + action_frames - 1
    return {
        "prefix_latents": prefix_latents,
        "prefix_pixel_frames": prefix_pixel_frames,
        "action_frames": action_frames,
        "model_frames": model_frames,
        "model_latents": 1 + (model_frames - 1) // _TEMPORAL_COMPRESSION,
    }


def prefix_skip_reason(source_len: int, timeline: dict[str, int], max_frames: int = MAX_FRAMES) -> str | None:
    """Why this (source, k) pair cannot be used, or None if it is runnable.

    Mirrors the training-side ``conditioning_max_frac=0.5`` guard on the COMBINED
    timeline and the VAE's max length. Never pads or splices earlier chunks: an
    insufficient tail is a skip, not a fabricated prefix.
    """
    prefix_pixel_frames = timeline["prefix_pixel_frames"]
    prefix_latents = timeline["prefix_latents"]
    if source_len < prefix_pixel_frames:
        return f"source has {source_len} frames, needs {prefix_pixel_frames}"
    if prefix_latents > timeline["model_latents"] // 2:
        return f"k={prefix_latents} exceeds half of the {timeline['model_latents']}-latent model timeline"
    if timeline["model_frames"] > max_frames:
        return f"model timeline {timeline['model_frames']} exceeds {max_frames} frames"
    return None


def eligible_prefix_latents(
    source_len: int,
    action_frames: int,
    candidate_ks=ALLOWED_PREFIX_LATENTS,
    max_frames: int = MAX_FRAMES,
) -> list[int]:
    """Subset of candidate k that this transition can actually supply."""
    eligible = []
    for k in candidate_ks:
        timeline = combined_timeline(action_frames, k)
        if prefix_skip_reason(source_len, timeline, max_frames) is None:
            eligible.append(k)
    return eligible
