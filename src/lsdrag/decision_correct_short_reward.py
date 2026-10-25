"""Optional exact-correct shortness bonus; does not change the existing length term.

Proposal only: active coefficient10 experiment does not import this module.
T is all generated argument/final tokens in the trajectory, not final answer alone.
"""
import math

def correct_short_reward(em, f1, generated_tokens, format_valid, *, length_weight=10.0, correct_short_weight=0.5, scale=256.0):
    if em not in (0, 1) or not math.isfinite(f1) or not 0 <= f1 <= 1:
        raise ValueError('Expected exact-match indicator and F1 in [0,1]')
    if isinstance(generated_tokens, bool) or not isinstance(generated_tokens, int) or generated_tokens < 0:
        raise ValueError('Expected nonnegative integer generated token count')
    if not all(math.isfinite(v) for v in (length_weight, correct_short_weight, scale)) or min(length_weight, correct_short_weight) < 0 or scale <= 0:
        raise ValueError('Invalid reward coefficients')
    length_penalty = length_weight * generated_tokens / (generated_tokens + scale)
    correct_short_bonus = correct_short_weight * em * scale / (generated_tokens + scale)
    base_reward = em + 0.25 * f1 - length_penalty + 0.02 * bool(format_valid)
    return dict(reward=base_reward + correct_short_bonus, base_reward=base_reward,
                length_penalty=length_penalty, correct_short_bonus=correct_short_bonus)
