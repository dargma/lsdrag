"""Training-only binary cost-to-go targets under an aggregate F1 floor."""
from collections import Counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def budget_targets(records):
    counts = Counter(r['qid'] for r in records)
    weight = np.array([1 / len(counts) / counts[r['qid']] for r in records])
    anchor = np.array([int(np.argmax(r['reference_probs']) == 3) for r in records])
    quality, cost, lower, upper = [], [], [], []
    for record, choice in zip(records, anchor):
        samples = {s['action']: s for s in record['samples']}
        assert len(samples) == len(record['samples']) == 2
        cont, stop = samples[record['continue_action']], samples[3]
        q = [cont['f1'], stop['f1']]
        c = [cont['generated'], stop['generated']]
        quality.append(q)
        cost.append(c)
        lo, hi = 0, 1
        if cont['errors'] or stop['errors'] or max(q) == 0 or (q[0] == q[1] and c[0] == c[1]):
            lo = hi = choice
        elif q[choice] > 0:
            # Never trade a reference-positive answer for a zero-F1 answer.
            if q[0] == 0:
                lo = 1
            if q[1] == 0:
                hi = 0
        lower.append(lo)
        upper.append(hi)
    quality, cost = np.array(quality), np.array(cost)
    dq, dc = weight * (quality[:, 1] - quality[:, 0]), weight * (cost[:, 1] - cost[:, 0])
    q0, c0 = weight @ quality[:, 0], weight @ cost[:, 0]
    floor = q0 + dq @ anchor
    baseline_cost = c0 + dc @ anchor
    bounds = Bounds(lower, upper)

    def solve(objective, rows, lows, highs):
        result = milp(objective, integrality=np.ones(len(records)), bounds=bounds,
                      constraints=LinearConstraint(np.array(rows), lows, highs),
                      options={'time_limit': 30})
        assert result.success, result.message
        selected = np.rint(result.x).astype(int)
        assert np.max(np.abs(result.x - selected)) < 1e-5
        return selected

    # Lexicographic training oracle: cost, quality, then minimum policy changes.
    rows, lows, highs = [dq], [floor - q0 - 1e-9], [np.inf]
    selected = solve(dc, rows, lows, highs)
    best_cost = dc @ selected
    rows.append(dc); lows.append(-np.inf); highs.append(best_cost + 1e-7)
    selected = solve(-dq, rows, lows, highs)
    best_quality = dq @ selected
    lows[0] = best_quality - 1e-9
    selected = solve(weight * (1 - 2 * anchor), rows, lows, highs)
    achieved_quality, achieved_cost = q0 + dq @ selected, c0 + dc @ selected
    assert achieved_quality >= floor - 1e-7
    assert achieved_cost <= baseline_cost + 1e-6
    labels = {i: int(value) for i, value in enumerate(selected) if value != anchor[i]}
    evidence = dict(states=len(records), questions=len(counts), changed_states=len(labels),
                    changed_questions=len({records[i]['qid'] for i in labels}),
                    reference_suffix_f1=float(floor), target_suffix_f1=float(achieved_quality),
                    reference_suffix_generated=float(baseline_cost),
                    target_suffix_generated=float(achieved_cost),
                    note='Binary hindsight targets on training prefixes; overlapping suffixes '
                         'are a weighted surrogate, not episode cost or deployable oracle performance.')
    return labels, evidence
