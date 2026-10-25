# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prediction text parsing and terminal-step detection.

Kept separate from the engines so the exact rules the rollout uses to read a
completion can be tested on stored model output — no model required.
"""

from __future__ import annotations

import re

_TAG_RE = {
    "subplan": re.compile(r"<subplan>(.*?)</subplan>", re.DOTALL),
    "world_modeling": re.compile(r"<world_modeling>(.*?)</world_modeling>", re.DOTALL),
    "duration": re.compile(r"<duration>(.*?)</duration>", re.DOTALL),
}

# Training completion markers: stop before generating an additional action chunk.
DONE_REPLACEMENTS = (
    "The task has been completed successfully.",
    "This task is now finished and ready for review.",
    "The requested work has been completed.",
    "All work has been done as instructed.",
)
_NORMALIZED_DONE = tuple(re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip() for text in DONE_REPLACEMENTS)


def is_done_instruction(instruction: object) -> bool:
    """A *labelled* terminal step (case and punctuation insensitive)."""
    if not isinstance(instruction, str):
        return False
    normalized = re.sub(r"[^a-z0-9]+", " ", instruction.casefold()).strip()
    return any(done in normalized for done in _NORMALIZED_DONE)


def parse_duration_seconds(value) -> float | None:
    """First number inside a duration field ("4s", "about 4.5 seconds", 4)."""
    if value is None:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


def parse_interleaved_tag_prediction(text: str) -> dict[str, str | bool | None]:
    """Parse ``<subplan>/<world_modeling>/<duration>`` independently, per field.

    Deliberately not an all-or-nothing parse: the rollout needs to degrade
    gracefully. A missing ``<subplan>`` or ``<world_modeling>`` keeps the complete
    raw completion for that field (so the Generator still gets a runnable
    instruction), while ``<duration>`` has no text fallback because it must be
    numeric before it can drive the frame count.
    """
    raw_text = text.strip()
    spans = {
        key: (match.group(1).strip() if match else None)
        for key, match in ((key, regex.search(text)) for key, regex in _TAG_RE.items())
    }
    subplan_ok = bool(spans["subplan"])
    world_modeling_ok = bool(spans["world_modeling"])
    duration_ok = bool(spans["duration"])
    any_tag_ok = subplan_ok or world_modeling_ok or duration_ok
    return {
        "format": (
            "tags"
            if subplan_ok and world_modeling_ok and duration_ok
            else "partial_tags"
            if any_tag_ok
            else "none"
        ),
        "raw_text": raw_text,
        "tags_format_ok": subplan_ok and world_modeling_ok and duration_ok,
        "subplan": spans["subplan"] if subplan_ok else raw_text,
        "world_modeling": spans["world_modeling"] if world_modeling_ok else raw_text,
        "duration": spans["duration"],
        "subplan_tag_parse_ok": subplan_ok,
        "world_modeling_tag_parse_ok": world_modeling_ok,
        "duration_tag_parse_ok": duration_ok,
    }
