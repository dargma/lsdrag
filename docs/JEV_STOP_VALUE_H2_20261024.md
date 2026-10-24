# H2: learn when further exploration is worthwhile

Hypothesis: H1 already has useful tool choices, but wastes generation on some
continuations. A state-conditioned stopping adjustment can retain its quality
while avoiding unnecessary execution. One candidate, no architecture grid.

## Fast matched evidence

Use the remaining48 of frozen train192, disjoint from all three earlier48
batches. Freeze hash order with seed20261067. H1 checkpoint32849a28 is the
collector and suffix policy. Same Qwen/native interface/shared101962 corpus.
At the initial and last nonforced state, execute FINAL and the best legal
non-FINAL action according to H1, then finish under frozen H1.

Roll-in and suffix are greedy, matching deployment, not stochastic two-sample
estimates. One execution per action is sufficient to avoid identical greedy
repeats; these are deterministic local policy counterfactuals, not confidence
intervals or optimal Q values. Up to240episodes and1800seconds after load,
generated500k/input30M. Local-fast only; audit/archive before fit.

## One stopping policy candidate

Freeze H1 and its normalization. Add a zero-initialized linear scalar to the
FINAL logit, using its existing128-dimensional MLP hidden. The three tool logits
and their relative ranking remain unchanged at the same state. A changed
trajectory may still expose different future states. No Native fallback,
backbone training, tool-format change or mechanically shortened episode cap.

Training labels use which of FINAL/continuation actually improves F1, or saves
generation at equal positive F1. Both-zero outcomes are ambiguous, never cheap
stopping targets. Errored pairs are excluded. Require each label across at least
three distinct training questions; otherwise do not train on unsupported labels.
Old training features provide current-H1 behavior retention, not old Q labels.

Use100 CPU updates/AdamW1e-3/seed20261068, equal correction/retention group
weights, question-balanced samples, target .5 reference stop probability +
.5 supported stop/continue label. Keep the final checkpoint, no epoch search.
This scalar is a learned stopping preference, not a calibrated value estimate.
Fitting does not prove useful policy improvement.

## Judgment

After verification, compare H1 and the single candidate on development64
(128episodes,1800seconds). Report F1, generated tokens, and whether useful
correct-answer-path savings survive instead of merely abandoning failures.
Use the existing fixed basic GRPO/native references. No extra ablation matrix.
Positive direction warrants expansion/independent confirmation; negative result
selects the next large hypothesis. Do not tune on particular dev failures.
Reserved80 stays untouched and goal remains ACTIVE until independent joint
success and useful policy evidence are established.
