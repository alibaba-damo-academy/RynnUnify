"""Training rewards for numerical, spatial and textual answers."""

import math
import re
from .legacy import standalone_scoring as scoring, mess_scoring
from .legacy.core import deterministic_match
from .legacy.sharp_reward import curve


class FeedbackError(RuntimeError):
    """Missing technical feedback. Abort/retry; never turn into zero reward."""


def phys_error(candidate, reference):
    # Strict numeric parser. Bare predictions inherit the reference unit.
    pattern = r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(cm/s|m/s|mm/s|cm|mm|m|ms|s)?\s*"
    units = {
        "cm/s": ("speed", 0.01),
        "m/s": ("speed", 1),
        "mm/s": ("speed", 0.001),
        "cm": ("length", 0.01),
        "mm": ("length", 0.001),
        "m": ("length", 1),
        "s": ("time", 1),
        "ms": ("time", 0.001),
    }
    gt = re.fullmatch(pattern, reference)
    pred = re.fullmatch(pattern, candidate)
    if not gt or gt[2] not in units or not math.isfinite(float(gt[1])):
        raise ValueError("Phys reference requires a finite number and supported unit")
    if not pred or not math.isfinite(float(pred[1])):
        return None
    pu = pred[2] or gt[2]
    if units[pu][0] != units[gt[2]][0]:
        return None
    p, g = float(pred[1]) * units[pu][1], float(gt[1]) * units[gt[2]][1]
    return abs(p - g) / abs(g) if g else (0.0 if p == g else None)


class Reward:
    def __init__(self, judge=None):
        self.judge = judge

    async def __call__(self, question, candidate):
        task, gt = question.task, question.reference
        if task == "choice":
            g = scoring.extract_choice(gt)
            if g is None:
                raise ValueError("Choice reference is not A/B/C/D")
            return float(scoring.extract_choice(candidate) == g)
        if task == "phys":
            return curve(phys_error(candidate, gt))
        if task == "distance":
            return curve(scoring.distance_metrics(candidate, gt)["relative_error"])
        if task == "point":
            if not scoring.parse_points(gt):
                raise ValueError("Invalid point reference")
            result = scoring.point_metrics(candidate, gt)
            return curve(result["distance"] / 1000 if result["parsed"] else None)
        if task == "trajectory":
            if not scoring.trajectory_metrics(gt, gt)["parsed"]:
                raise ValueError("Invalid trajectory reference")
            return curve(scoring.trajectory_metrics(candidate, gt).get("relative_error"))
        if task == "text" and deterministic_match(candidate, gt):
            return 1.0
        if not candidate.strip():
            return 0.0
        if self.judge is None:
            raise FeedbackError("Text/Mess scoring needs a configured judge")
        messages = (
            mess_scoring.mess_judge_messages(question.question, gt, candidate)
            if task == "mess"
            else scoring.judge_messages(question.question, gt, candidate)
        )
        # Retry verdict parsing on the SAME answer; do not generate a new answer.
        for attempt in range(3):
            try:
                text = await self.judge(messages)
                if task == "mess":
                    return mess_scoring.parse_mess_verdict(text)["score"] / 100
                return scoring.parse_verdict(text) / 100
            except (FeedbackError, ValueError, TypeError):
                if attempt == 2:
                    raise FeedbackError("Judge did not return a valid complete verdict after 3 attempts") from None
