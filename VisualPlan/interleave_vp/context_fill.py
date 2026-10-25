# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Scene-memory context selection for ``--context-frames K``.

Keep the episode anchor in slot 0, spread the middle slots over older clips,
and reserve the final slot for the most recent clip. This preserves both
long-range scene references and short-range continuity.

When history is sparse, fill unused slots with interior frames from available
history clips. With no history, repeat the anchor rather than sample future
frames. The resulting context always contains K items in chronological order.
"""


def uniform_pick(num_candidates: int, m: int) -> list[int]:
    """``m`` evenly spread indices into a ``num_candidates``-long list.

    Interior-biased, so the picks sit inside the range rather than clinging to
    its ends. Returns fewer than ``m`` indices (ascending, unique) when the
    candidate list is too short.
    """
    if num_candidates <= 0 or m <= 0:
        return []
    out: list[int] = []
    for j in range(m):
        idx = int((j + 1) * (num_candidates + 1) / (m + 1) + 0.5) - 1
        idx = max(0, min(num_candidates - 1, idx))
        if idx not in out:
            out.append(idx)
    return sorted(out)


def allocate_fill(num_units: int, n_fill: int) -> list[int]:
    """Spread ``n_fill`` extra frames over ``num_units`` history units.

    Round-robin from the most recent unit backwards, so a single extra frame
    lands in the newest clip rather than the oldest.
    """
    if num_units <= 0:
        return []
    alloc = [0] * num_units
    for t in range(max(0, n_fill)):
        alloc[num_units - 1 - (t % num_units)] += 1
    return alloc


def interior_indices(frame_count: int, m: int) -> list[int]:
    """``m`` evenly spaced interior frame indices of a ``frame_count`` clip.

    Excludes frame 0, which is already that clip's keyframe. Returns fewer than
    ``m`` entries when the clip is too short for that many distinct frames.
    """
    if m <= 0 or frame_count <= 1:
        return []
    out: list[int] = []
    for j in range(m):
        idx = int(frame_count * (j + 1) / (m + 1) + 0.5)
        idx = max(1, min(frame_count - 1, idx))
        if out and idx <= out[-1]:
            idx = out[-1] + 1
        if idx > frame_count - 1:
            break
        out.append(idx)
    return out


def build_fill_plan(unit_lengths: list[int], n_fill: int) -> list[tuple[int, int]]:
    """Plan the fill frames as ``[(unit_index, frame_index), ...]``, time-ordered.

    Merging this with the prefill keyframes under the same ``(unit, frame)`` key
    yields a chronological context list. Training keyframes sit at ``(unit, 0)``;
    rollout keyframes sit at ``(unit, len-1)``.
    """
    plan: list[tuple[int, int]] = []
    for u, m in enumerate(allocate_fill(len(unit_lengths), n_fill)):
        for idx in interior_indices(unit_lengths[u], m):
            plan.append((u, idx))
    plan.sort()
    return plan


def select_history_units(num_history: int, k: int) -> list[int]:
    """Pick which history units fill the K-1 non-anchor slots.

    ``num_history`` counts the units available to this sample: on the training
    side the previous clips ``clips[0 .. i-1]`` (so ``num_history == i``), on the
    rollout side the generated chunks so far. Unit 0 doubles as the anchor on the
    training side, hence the dedup.

    Returns ascending unit indices: the most recent unit, plus ``k-2`` units
    spread evenly over everything between the anchor and it.
    """
    if num_history <= 0:
        return []
    most_recent = num_history - 1
    if most_recent == 0:
        return [0]  # only the anchor's own clip is available
    # Candidates strictly between the anchor (unit 0) and the most recent unit.
    candidates = list(range(1, most_recent))
    picks = [candidates[c] for c in uniform_pick(len(candidates), max(0, k - 2))]
    return sorted(set(picks + [most_recent]))
