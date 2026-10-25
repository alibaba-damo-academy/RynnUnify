"""Visual reasoning interaction contract."""

import math
import re

SYSTEM = """Answer the question from its visual observations. First choose <route>D</route> for direct reasoning or <route>G</route> to request additional visual evidence.
After D, output <thinking>your reasoning</thinking><answer>your final answer</answer>.
After G, output <world_modeling>a concrete visual generation instruction</world_modeling><duration>4s</duration>, then stop. Duration must be greater than 0 and at most 10 seconds. Describe relevant visible objects, motion, support/contact and the requested interaction. After receiving the generated video, output <thinking>your reasoning</thinking><answer>your final answer</answer>. Do not output other text."""


def answer(text):
    match = re.fullmatch(r"\s*<thinking>(.*?)</thinking>\s*<answer>(.*?)</answer>\s*", text, re.S)
    if (
        match
        and match[2].strip()
        and all(text.count(tag) == 1 for tag in ("<thinking>", "</thinking>", "<answer>", "</answer>"))
    ):
        return match[2].strip()
    return None


def request(text):
    match = re.fullmatch(
        r"\s*<world_modeling>(.*?)</world_modeling>\s*<duration>(\d+(?:\.\d+)?)s</duration>\s*", text, re.S
    )
    if not match or not match[1].strip():
        return None
    if any(text.count(tag) != 1 for tag in ("<world_modeling>", "</world_modeling>", "<duration>", "</duration>")):
        return None
    duration = float(match[2])
    return (match[1].strip(), duration) if math.isfinite(duration) and 0 < duration <= 10 else None


def messages(question):
    # Whitelist model-visible fields. Never serialize the whole training row.
    content = [{"type": "image", "image": p} for p in question.images]
    content += [{"type": "video", "video": p} for p in question.videos]
    content += [{"type": "text", "text": question.question}]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
