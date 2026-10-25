#!/usr/bin/env python3
"""Duration <-> frame-count mapping for labels and interleaved inference.

Single source of truth for the quantization chain:

  label:  gt frame_count -> clamp [MIN_FRAMES, MAX_FRAMES] -> floor to 4n+1
          -> / fps -> round to whole seconds -> min 1s
  infer:  predicted whole seconds d -> round(d * fps) frames -> floor to 4n+1
          -> clamp [MIN_FRAMES, MAX_FRAMES]

The Generator VAE requires 4n+1 pixel frames (temporal compression 4, latent 0
is the first pixel frame alone). Keeping both directions here guarantees the
training labels and the inference mapping never drift apart.
"""

from __future__ import annotations

MIN_FRAMES = 5  # t_latent >= 2
MAX_FRAMES = 241  # maximum supported clip length
DEFAULT_FPS = 24
MIN_DURATION_S = 1


def snap_frames_4n1(num_frames: int, max_frames: int = MAX_FRAMES, min_frames: int = MIN_FRAMES) -> int:
    """Clamp to [min_frames, max_frames] then floor to the 4n+1 grid."""
    requested = max(min_frames, min(int(num_frames), max_frames))
    return (requested - 1) // 4 * 4 + 1


def duration_label_seconds(frame_count: int, fps: int = DEFAULT_FPS, max_frames: int = MAX_FRAMES) -> int:
    """Whole-second Reasoner label for a GT clip: round(snapped_frames / fps), min 1s."""
    snapped = snap_frames_4n1(frame_count, max_frames=max_frames)
    return max(MIN_DURATION_S, int(round(snapped / fps)))


def frames_from_duration(duration_s: float, fps: int = DEFAULT_FPS, max_frames: int = MAX_FRAMES) -> int:
    """Map a predicted whole-second duration back to a 4n+1 generator frame count."""
    raw = int(round(float(duration_s) * fps))
    return snap_frames_4n1(raw, max_frames=max_frames)
