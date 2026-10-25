# Task rewards and judge settings

For protocol-governed questions, the final reward is `format_valid * answer_score`. There is no fixed bonus for format or for calling the generator. A malformed model response gets zero; a service failure raises `FeedbackError` and aborts the batch. Native multi-image choice tasks use their answer score independently of output format and have no routing loss.

Continuous tasks use `C(e) = max(0, 1 - e/0.30)^3`. This is a shaping reward, not an accuracy percentage or an official benchmark metric.

| Task key | Answer score | Implementation/source |
|---|---|---|
| `choice` | Exact A/B/C/D extraction and equality | choice parser |
| `phys` | C(abs(pred-gt)/abs(gt)) after unit conversion | Strict signed parser |
| `distance` | C(relative distance error) | meter parser and distance metrics |
| `point` | C(symmetric mean nearest-point distance / 1000) | point scorer; covers affordance/area |
| `trajectory` | C(discrete Fréchet distance / GT arc length) | scorer; first 10 points, 15 arc-length samples |
| `text` | Deterministic normalized match, otherwise binary judge | matching and judge examples |
| `mess` | Level / 4, levels 0 through 4 | `mess_levels_v1` rubric |

Phys accepts signed finite scalars and cm, mm, m, cm/s, mm/s, m/s, s, ms. A bare prediction inherits the reference unit. Incompatible units or ambiguous prose give zero. A zero reference earns one only on exact zero after conversion; other values earn zero.

Point coordinates use the <=2 normalization heuristic; values near the origin can be interpreted as normalized coordinates. Area is point-set distance, not IoU. Trajectory order is retained. A zero-length GT trajectory earns full credit only when its discrete Fréchet distance is exactly zero.

Text judging uses the accuracy/completeness prompt with examples and requires exactly `###Judge: True` or `###Judge: False`. Mess judging assesses affected non-manipulated objects, direction, magnitude, rotation and support/contact effects, returning a JSON `level` and justification. See `rynn_rl/legacy/mess_scoring.py` for the full rubric. The judge receives question, reference and final answer; it receives neither the reasoning transcript nor the generated video.

The public judge client uses temperature 0, 512 maximum output tokens, a 120-second timeout, and up to three attempts with the same answer. Truncated or malformed verdicts are missing feedback, not incorrect answers. Judge model, endpoint and authentication are configurable. Run comparisons with a fixed judge checkpoint and rubric.

