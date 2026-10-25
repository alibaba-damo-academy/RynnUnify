"""Task scoring functions and synthetic grading examples."""
import math
import re
PAIR_RE = re.compile(r"[(\[]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[)\]]")
TRAJECTORY_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
def extract_answer(text):
    text = re.sub(r"<(thinking|think|world_modeling)>.*?(?:</\1>|$)", "", text or "", flags=re.I | re.S)
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    return (matches[-1] if matches else text).strip()

def extract_choice(text):
    answer = extract_answer(text)
    bare = answer.strip("()[] .\n\t").upper()
    if bare in {"A", "B", "C", "D"}:
        return bare
    matches = set(re.findall(r"\b[A-D]\b", answer))
    return matches.pop() if len(matches) == 1 else None

def parse_points(text):
    tagged = re.findall(r"<(?:affordance|area)>(.*?)</(?:affordance|area)>", text, re.I | re.S)
    pairs = [(float(x), float(y)) for x, y in PAIR_RE.findall(tagged[-1] if tagged else text)]
    if any(not math.isfinite(v) for point in pairs for v in point):
        return []
    # Deliberately retains the supplied reward's <=2 heuristic, even near the origin.
    if pairs and max(abs(v) for point in pairs for v in point) <= 2:
        pairs = [(x * 1000, y * 1000) for x, y in pairs]
    return pairs

def point_metrics(pred, gt, scale=5.0, tolerance=50.0):
    pred, gt = parse_points(pred), parse_points(gt)
    if not pred or not gt:
        return {"score": 0.0, "distance": None, "hit": 0.0, "parsed": False}

    def nearest(a, b):
        return sum(min(math.hypot(x - u, y - v) for u, v in b) for x, y in a) / len(a)

    distance = (nearest(pred, gt) + nearest(gt, pred)) / 2
    return {"score": 100 * math.exp(-distance / 1000 * scale), "distance": distance,
            "hit": float(distance <= tolerance), "parsed": True}

def resample(points, n=15):
    # Arc-length interpolation; collapse zero-length segments without changing the path.
    clean = [points[0]]
    cumulative = [0.0]
    for point in points[1:]:
        distance = math.hypot(point[0] - clean[-1][0], point[1] - clean[-1][1])
        if distance:
            clean.append(point)
            cumulative.append(cumulative[-1] + distance)
    if len(clean) == 1:
        return clean * n
    result, j = [], 1
    for i in range(n):
        target = cumulative[-1] * i / (n - 1)
        while j < len(clean) - 1 and cumulative[j] < target:
            j += 1
        weight = (target - cumulative[j - 1]) / (cumulative[j] - cumulative[j - 1])
        result.append(tuple(a + (b - a) * weight for a, b in zip(clean[j - 1], clean[j])))
    return result

def trajectory_metrics(pred, gt):
    sequences = [[(float(x) / 1000, float(y) / 1000)
                  for x, y in TRAJECTORY_RE.findall(text)[:10]] for text in (pred, gt)]
    if any(not seq or any(not math.isfinite(v) for p in seq for v in p) for seq in sequences):
        return {'score': 0., 'acc_hit': 0., 'parsed': False, 'dfd': None, 'gt_length': None}
    gt_length = sum(math.dist(u, v) for u, v in zip(sequences[1], sequences[1][1:]))
    a, b = [resample(seq) for seq in sequences]
    costs = [[0.0] * len(b) for _ in a]
    for i, (x, y) in enumerate(a):
        for j, (u, v) in enumerate(b):
            distance = math.hypot(x - u, y - v)
            if i == j == 0:
                costs[i][j] = distance
            else:
                previous = ([costs[i - 1][j]] if i else []) + ([costs[i][j - 1]] if j else [])
                if i and j:
                    previous.append(costs[i - 1][j - 1])
                costs[i][j] = max(min(previous), distance)
    distance = costs[-1][-1]
    return {'score': 100 * math.exp(-distance), 'dfd': distance, 'gt_length': gt_length,
            'relative_error': distance / gt_length if gt_length else (0. if distance == 0 else None),
            'acc_hit': float((distance < .1 * gt_length and not math.isclose(distance, .1 * gt_length, rel_tol=1e-12, abs_tol=0.)) if gt_length else distance == 0),
            'parsed': True, 'hit_contract': 'dfd_lt10pct_gt_length_v1'}

def trajectory_score(pred, gt):
    return trajectory_metrics(pred, gt)['score']

def judge_messages(question, gt, pred):
    # Binary accuracy/completeness prompt with synthetic grading examples.
    # Synthetic examples illustrate verdict format and completeness.
    examples = [('What color is the cube?', 'Blue.', 'The cube is blue.', True), ('Which objects move?', 'The sphere and the cube.', 'The sphere.', False), ('Does the cylinder fall?', 'No.', 'The cylinder falls.', False), ('Which direction does the sphere move?', 'Left.', 'It moves to the left.', True)]
    system = (
        "You are a helpful assistant. Please judge whether the predicted answer is correct or not according to the given question and labeled answer. "
        "You need to consider the answer from two perspectives: accuracy and completeness. "
        "Output ###Judge: True only when the predicted answer is accurate and complete; otherwise, output ###Judge: False"
    )
    messages = [{"role": "system", "content": system}]
    for q, label, answer, verdict in examples:
        messages.extend([
            {"role": "user", "content": f"###Question: {q} ###Label Answer: {label} ###Predicted Answer: {answer}"},
            {"role": "assistant", "content": f"###Judge: {verdict}"},
        ])
    messages.append({"role": "user", "content": f"###Question: {question} ###Label Answer: {gt} ###Predicted Answer: {pred}"})
    return messages

def parse_verdict(text):
    match = re.fullmatch(r"\s*###Judge:\s*(True|False)\s*", text, re.I)
    if not match:
        raise ValueError("Malformed GPT verdict: expected exactly ###Judge: True or ###Judge: False")
    return 100.0 if match[1].lower() == "true" else 0.0

DISTANCE_SCORER = "distance_relative_linear_v1"
def parse_distance_meters(text):
    """Parse one unambiguous length, not object IDs, ranges or qualitative labels."""
    number = r"[+-]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    units = {
        "m": 1, "meter": 1, "meters": 1, "metre": 1, "metres": 1, "米": 1,
        "cm": .01, "centimeter": .01, "centimeters": .01, "centimetre": .01,
        "centimetres": .01, "厘米": .01,
        "mm": .001, "millimeter": .001, "millimeters": .001, "millimetre": .001,
        "millimetres": .001, "毫米": .001,
        "km": 1000, "kilometer": 1000, "kilometers": 1000, "kilometre": 1000,
        "kilometres": 1000, "公里": 1000, "千米": 1000,
        "ft": .3048, "foot": .3048, "feet": .3048,
        "in": .0254, "inch": .0254, "inches": .0254,
        "yd": .9144, "yard": .9144, "yards": .9144,
    }
    text = extract_answer(text).strip()
    if re.search(r"(?:/|\bper\b)\s*(?:s\b|sec|second|minute|hour)|[²³]", text, re.I):
        return None  # A speed/area/volume is not a distance.
    # Object identifiers do not count as candidate distance estimates.
    text = re.sub(r"<?\bobject\s*\d+\b>?", "", text, flags=re.I)
    unit_pattern = "|".join(sorted(units, key=len, reverse=True))
    pattern = re.compile(rf"(?<![\w.,])({number})\s*({unit_pattern})(?!\w)", re.I)
    matches = list(pattern.finditer(text))
    if matches:
        # An unmatched number indicates a range, alternative, coordinate, or
        # unsupported mixed-unit expression. Never silently take its last number.
        if re.search(r"\d", pattern.sub("", text)):
            return None
        values = [float(m[1].replace(",", "")) * units[m[2].lower()] for m in matches]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            return None
        if not all(math.isclose(v, values[0], rel_tol=1e-9, abs_tol=1e-9) for v in values):
            return None
        return values[0]
    # The prompt explicitly specifies meters, so a bare scalar is unambiguous.
    bare = re.fullmatch(rf"(?:about\s+|approximately\s+|approx\.?\s+|[~≈约]\s*)?({number})[。]?", text, re.I)
    if bare:
        value = float(bare[1].replace(",", ""))
        return value if math.isfinite(value) and value >= 0 else None
    return None

def distance_metrics(prediction, ground_truth):
    gt = parse_distance_meters(ground_truth)
    if gt is None:
        raise ValueError("Spatial-distance reference is not an unambiguous non-negative length")
    pred = parse_distance_meters(prediction)
    result = {"scorer": DISTANCE_SCORER, "score": 0.0, "parsed": pred is not None,
              "prediction_m": pred, "ground_truth_m": gt, "absolute_error_m": None,
              "relative_error": None, "acc_10": False, "acc_20": False}
    if pred is None:
        return dict(result, parse_error="Expected one numerical distance; vague or ambiguous answer")
    error = abs(pred - gt)
    relative = error / gt if gt else (0.0 if error == 0 else None)
    if relative is not None and not math.isfinite(relative):
        relative = None
    result.update(absolute_error_m=error, relative_error=relative,
                  score=100.0 * max(0.0, 1.0 - relative) if relative is not None else 0.0,
                  acc_10=relative is not None and relative < .10 and not math.isclose(relative, .10, rel_tol=1e-12, abs_tol=0.),
                  acc_20=relative is not None and relative <= .20 + 1e-12)
    return result
