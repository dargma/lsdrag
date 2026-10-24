> Latest user direction: hypothesis-first, fast single-candidate policy tests; detailed ablations only after direction narrows. Active strategy: `JEV_DECISION_POLICY_HYPOTHESIS_PLAN_20261024.md`. Third48 collection is now complete and independently verified; no new candidate has been fitted yet. Historical runtime descriptions below are preserved as history.

# Current experiment and MLP lineage after runtime recovery

This map resolves historical execution pointers; actual process/artifact checks
and the newest memory headers remain necessary. User requested latest recovery,
explicit goal mode, and clarification of the MLP experiments. Remote jev_v1.1
at pre-recovery commit0a310f2 (09:11UTC) contains jev_v1.2 at9a9ef5b (04:11UTC)
as an ancestor. Recovery did not revert to the older MLP experiment.

## Goal and fixed environment

F1 +5 percentage points AND generated-token ratio <=0.70 against fixed basic
shared-corpus GRPO59a0f766, plus evidence of useful state-conditioned action
selection. Exposed validation64 is development, never independent confirmation.
Freeze candidate/confirmation protocol before reserved80 access. Native A-RAG
interface, frozen nonthinking Qwen4B, shared101962 corpus and train192 are unchanged.

## Main completed development comparisons

All rows below are the same exposed64 questions. Objectives, collection policies
and training datasets differ; this is a history of candidates, not a controlled
architecture-only comparison. Scores are F1 percent and average generated tokens.

| Policy | F1 | Generated |
|---|---:|---:|
| Fixed basic GRPO |22.50|98.41|
| Native fast |24.70|97.44|
| Selected multi-position MLP d987 |25.18|76.75|
| d987 + GRPO |22.95|53.19|
| Zero-quality-group-gated GRPO |25.96|83.63|
| Auxiliary quality/cost MLP |32.45|139.03|
| Quality MLP + GRPO |34.75|150.91|
| Fresh matched-target MLP |23.36|77.66|
| Fresh matched-target frozen question attention |21.44|64.86|
| Direct predicted quality/cost selector |12.02|75.64|
| Selective quality-correction MLP f07f |26.41|81.55|
| Selective quality-correction frozen question attention |25.18|78.02|
| Cost-priority correction MLP e1a630e5 |26.74|75.45|

Exact development thresholds: F1 >=27.5006764, generated <=68.884375. None of
these candidates passes both. No research-goal success is established.

Sources: `JEV_QUALITY_HEAD_GRPO_RESULT_20261024.md`,
`JEV_ZERO_GATE_RESULT_20261024.md`, `JEV_MATCHED_REFRESH_RESULT_20261024.md`,
`JEV_ANCHOR_CORRECTION_RESULT_20261024.md`,
`JEV_COST_PRIORITY_RESULT_20261024.md`, and restored raw-reference verification
`lab/registry/recovery_20261024/followup_verification.json`.

## Current branch, not the highest-F1 branch

The current efficient offline branch is d987 -> selective quality correction
f07f -> cost-priority correction e1a630e5. The high-F1/high-cost auxiliary MLP
and its GRPO descendant remain preserved as separate branches. They are not
the policy currently collecting data.

Current head: three2560-dimensional state summaries (last token, question-end,
last-observation mean), fixed normalization, Linear7680->128, Tanh,
Linear128->4, legal-action masking. Only action selection uses this MLP;
the frozen LLM generates tool arguments and final answers.

The latest cost-priority result's net token reduction was dominated by four
still-incorrect questions stopping without tools. Therefore it does not establish
more efficient correct-answer paths. A completed training-state audit shows
18/21 correction targets adopted, all21 target probabilities increased, and
0/66 uncorrected fresh-state action changes. This is in-sample fitting evidence,
not proof of a generalization bottleneck or of policy understanding.

## Current collection and next decision

Namespace: `decision_cost_priority_refresh48c_20261024`.
Protocol: `JEV_COST_POLICY_COVERAGE_PROTOCOL_20261024.md`.
48 questions disjoint from the earlier96, within unchanged train192;
seed20261065; initial/last nonforced state; two suffix samples per legal action;
frozen e1a630e5 roll-in/suffix. No PPO/GRPO training in this phase.

The reset lost unarchived partial results. Explicit user resume authorized
rerunning this collection with identical frozen inputs; completed earlier
experiments were restored, not rerun. Check coordinator state at
`/content/local_fast/lsdrag-text-v1/results/decision_cost_refresh_coordinator_20261024.json`.

After full completion: verify raw data, scorer, Drive archive and opportunity
categories; then freeze a bounded MLP correction fit if the complete train-only
evidence supports it. Zero-F1-to-zero-F1 savings are excluded as correction
targets. Preserve anchor behavior on other training features. Do not pool old
suffix outcomes as if generated under the current policy. Compare a frozen
candidate against current head on paired development64 and cached basic/native
references, including correct-versus-failed path cost contributions. The next fit
is not yet frozen in this map; no inference should be drawn from partial data.
