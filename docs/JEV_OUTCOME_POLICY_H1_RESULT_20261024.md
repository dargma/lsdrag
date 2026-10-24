# H1 result: useful policy changes, insufficient total cost reduction

One outcome-supported MLP candidate, unchanged native interface/frozen Qwen,
paired exposed development64.128episodes completed in192.12seconds; no errors.
Current-reference outputs/F1/support/tokens/calls/errors exactly replay all64.
Raw totals and Drive archive SHA independently checked.

| Policy | F1 (%) | Generated tokens |
|---|---:|---:|
| Fixed basic GRPO |22.50|98.41|
| Current cost-priority MLP |26.74|75.45|
| H1 outcome-supported MLP |28.30|75.34|

Versus current: +1.5625F1pp (one win, no losses), marginal paired95% bootstrap
CI[0,+4.6875]pp. Generated ratio0.99855: essentially unchanged, CI[.925,1.102].
Versus basic GRPO: +5.80F1pp and approximately23.44% fewer generated tokens.
The development quality threshold is crossed; the30% token target is not.
These are adaptive development results, single training seed, not independent
confirmation or robust improvement established from one recovered answer.

The main hypothesis has a useful signal: changing initial keyword search to
semantic search recovers one answer while saving2tokens. Two other questions
retain F1=1 while avoiding unneeded calls, saving55 and66tokens. Thus this update
has concrete correct-answer-path improvements, unlike the preceding update's
savings dominated by wrong early exits. This is limited paired evidence, not
proof the model understands future information value.

The cost benefit is canceled elsewhere: still-zero-F1 questions use116 more
tokens in aggregate, including one formerly immediate empty answer that now
searches/reads and still fails, adding177tokens. Net saving across64 is only7.
This is descriptive decomposition, not permission to identify failing questions
with gold labels at inference or to tune against their IDs.

Next large hypothesis: the policy needs a better estimate of whether continuing
will produce useful information. Retain the promising H1 candidate; prioritize
one train-only stopping/continuation-value experiment before adding new
architectures. No broad ablation grid. Freeze that followup separately; existing
third48 labels use the previous e1a630e5 suffix, not this new policy.

Goal ACTIVE. No reserved80 access. Evidence:
`lab/reports/decision_outcome_policy_h1_validation64_20261024/`.
Checkpoint32849a288d13e53c32bf6b0e950d8dea791e36e0d0aa41114376bcbe01f9658a.

Operational recovery: GPU evaluation completed successfully. Reporting initially
failed because two historical split manifests had not been restored after reset.
Restored their original frozen Git copies and reran reporting/archive only;
no model evaluation was repeated. The original reporting error is preserved in
coordinator state as `reporting_failure_recovered`.
