"""Opt-in compact-support reward; native scorers and observation hits stay intact."""
import math
from dataclasses import replace

CURVE = 'relative_cubic_cutoff30_v1'
CUTOFF = 0.30

def curve(error):
    if error is None:
        return 0.0  # Parsed zero-GT non-exact case has undefined relative error.
    if math.isnan(error) or error < 0:
        raise ValueError('relative error must be nonnegative and not NaN')
    if error >= CUTOFF:
        return 0.0
    return (1.0 - error / CUTOFF) ** 3

def reshape(result):
    if not result.valid:
        return result
    m = dict(result.metrics)
    if result.source == 'benchmark_point':
        error = m['distance'] / 1000 if m.get('parsed') else None
        basis = 'symmetric_nearest_point_distance/1000'
    elif result.source in ('legacy_phys', 'benchmark_distance', 'trajectory_dfd'):
        error = m.get('relative_error')
        basis = 'dfd/gt_arc_length' if result.source == 'trajectory_dfd' else 'abs(pred-gt)/abs(gt)'
    else:
        return result  # Binary/Judge/Mess rewards have no relative-error axis.
    score = curve(error)
    m.update(reward_curve=CURVE, reward_error=error, reward_error_basis=basis,
             native_answer_reward=result.score, shaped_answer_reward=score)
    return replace(result, score=score, metrics=m)
