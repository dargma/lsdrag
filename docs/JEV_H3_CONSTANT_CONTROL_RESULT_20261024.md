# H3 constant-residual control: budget match does not transfer

Train24 token-only calibration selected bias−0.0847940594. H3 generated67.0417
versus constant67.5833 (ratio1.00808), inside frozen10%tolerance. Selection used
no F1 and was frozen before dev. Only selected scalar evaluated on dev.

| Policy | Dev F1 (%) | Generated tokens |
|---|---:|---:|
| H3 state-dependent residual |27.32|66.89|
| H1 + constant FINAL offset |28.30|77.59|

Dev constant/H3 token ratio1.16001 exceeds the frozen10%gate. Thus this is NOT
a successful equal-budget mechanism comparison; do not claim H3's conditional
residual beats a constant at equal cost. Nor does it prove state dependence is
useless. Constant gains.9884F1pp but uses16.00%moretokens; paired F1difference
CI[−5.3149,+6.7708]pp. Constant versus H3:4wins/3losses. Both are useful cost/
quality reference points; neither independently establishes the original goal.

Constant costs685moretokens:383both-zero,143constant-better,33H3-better,
126positive-tied. H3reference replay64exact. Raw scores rechecked against
source answers; candidate tensors have zero residual weights, frozen H1base
matches H3; scalar selection recomputed on raw generated counts only; source/
input hashes and calibration/dev Drive archives byte-verified. Eval128episodes,
186.997seconds,9247generated/460319input,0errors. No dev retuning.

Latest user direction: real multi-turn policy improvement with a shared reader
backbone. Preserve H3 and this negative/inconclusive diagnostic. Move to one
bounded H3-current-policy all-action outcome collection and a single subsequent
policy update; no more constant sweeps, no new separate LLM, no premature
unfreezing. Exact action learning to be frozen using train-only evidence.
See docs/JEV_SHARED_BACKBONE_DECISION_RESEARCH_20261024.md.

GoalACTIVE, reserved80 untouched. Evidence:
lab/reports/decision_h3_constant_validation64_20261024/independent_verification.json.
