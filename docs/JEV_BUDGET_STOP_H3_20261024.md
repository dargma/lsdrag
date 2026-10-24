# H3: spend continuation tokens only within a training quality budget

H2 improves quality but spends more because its labels prefer any F1 gain.
Test one alternative: choose training stop/continue decisions jointly to minimize
generation while retaining the current H1 aggregate training-prefix F1.
Use the already verified greedy H1 pairs; no new collection or architecture grid.

Targets are binary optimization on train-only83prefixes: equal question weight,
equal state weight within question; minimize remaining generated tokens subject
to weighted F1>=H1's greedy selected outcomes. Break cost ties by higher quality,
then fewer action changes. Exclude errored/all-zero pairs from changes, and never
turn a reference-positive outcome into zero F1. Positive-to-positive tradeoffs
may be compensated across training states; this is explicitly different from
H2's local quality-first labels. No dev labels, IDs, penalties or thresholds enter
the optimization. The target is a privileged training surrogate; overlapping
suffixes do not constitute a global episode budget or prove learnable improvement.

Fit only changed targets, keeping H1 on other fresh/old training features.
Require changes across at least3training questions. Keep H2's129-parameter
stopping head, zero-initialized around H1,100updates/AdamW1e-3/seed20261069,
.5reference-target blend and equal correction/retention group weight. No epoch
or coefficient sweep. Tool logits/normalization/backbone stay frozen.

One paired development64 comparison against H1,128episodes/1800seconds;
cached basic GRPO/native references. Decide from actual F1 and generation,
including whether correct-answer savings survive. Preserve failures. If targets
cannot yield useful change, do not call the surrogate optimum a policy result.
Reserved80 remains protected. Goal remains independent joint F1+5pp/generated
ratio<=.70 plus useful policy evidence, not this surrogate's feasibility.

## Coverage gate and bounded middle-state followup

Initial83 training prefixes yield only1changed question, below3: no fit launched.
This only covers first/last states. Collect all54 intermediate nonforced states
from34 of the same48 stored H1 greedy rollins, without outcome-based selection.
Reuse exact public prefixes; run FINAL and best legal H1 continuation once each,
108new episodes maximum,1200seconds after loading. Frozen manifest
lab/registry/decision_stop_middle_h3_20261024/manifest.json records parent hashes.
Require prefix hashes, legal actions and original H1 probabilities to replay.
No new rollin cost is incurred; counts include only new suffix executions.
After raw audit, combine unique same-policy prefixes with original83 and repeat
the unchanged training feasibility gate before any fit. No test/dev collection.

Independent coverage audit rejected the first middle collection: inherited
state_snapshots only returns first/middle/last, yielding34 rather than54states.
Preserve its68episodes; supplement exactly20missing states (40new episodes)
with explicit public-prefix reconstruction. Frozen supplement manifest records
missing turns and prior tensor hash. Only combined full coverage can pass.
