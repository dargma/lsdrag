> Latest user override2026-10-24: native ARAG protocol unchanged; action-selection head substitution and policy learning only. Short-reference/ID serialization proposal withdrawn. Older interface-optimization suggestions below are superseded.

# Remaining adaptive program and head lineage

Latest user direction: apply GRPO to the F1 32.45 MLP, continue ideas/experiments until the real goal is achieved, and consider the system broadly rather than only reward coefficients. Keep the frozen reader LLM and head-only trainable-parameter objective.

## Priority order

1. Finish current-policy train48 branch collection already running; audit/archive. It is a fixed subset of unchanged train192, not a new train/test split.
2. Continue the actual quality-leading auxiliary MLP with GRPO; queued bounded192×4×1pilot. This is the next GPU training experiment explicitly requested. See JEV_QUALITY_HEAD_GRPO_20261024.md.
3. Use both GRPO results and refreshed branch outcomes to decide the next policy update. Compare quality/cost target reliability and current-policy versus oldBC suffix mismatch. Prefer evidence-supported quality-preserving action preferences or refreshed value estimates over broad stale-cache penalty sweeps. Exact method not frozen yet.
4. If state information is insufficient, expand what the head can read: actual observation-token features and action-conditioned soft attention, not only three pooled slots. This was discussed as a candidate; existing attention results only tested three summaries and do not rule out token-level inputs. Prepare a controlled same-data comparison only if warranted.
5. Inspect remaining bottlenecks across query arguments, repeated retrieval, read selection, stopping, final-answer generation, and dataset coverage. Distinguish policy errors from frozen-reader/retrieval limitations. Interface or generation-format changes require matched baselines and explicit protocol/source provenance; do not silently change retrieval or the evaluation task. No backbone weight training, world-model or memory expansion.

Avoid a full factorial grid. Combine shared data collection, reuse train caches and serialize single-GPU execution. Keep failed experiments and account for cost. Frozen independent confirmation is still required; exposed validation gains do not close the goal.

## Exact branches

Initial teacher action learning → selected multi-position MLP outcome learning (d987...):
- branchA: ordinary GRPO → validationF122.95/gen53.19.
- branchB: zero-quality-group-gated GRPO →25.96/83.63.
- branchC: further offline policy+stopping+quality/cost auxiliary →32.45/139.03 (0a1b...). No GRPO on this branch yet at plan creation.
  - constrained offline budget70/55 descendants failed to retain quality; preserve results.
  - NOW queued: fresh GRPO directly from0a1b..., not from constrained descendants.

The current refresh collector also uses frozen0a1b..., and its counterfactual branch data must not be presented as on-policy GRPO data. The branch structure matters: do not describe all MLP results as receiving GRPO.

## Train-only cost evidence and adaptive decision rule

Latest user reiterates autonomous broad investigation until the actual joint goal. Current exact quality-MLP GRPO remains first priority; do not change its frozen protocol mid-run.

Train48 stochastic collector accounting (not validation): 5272 generated tokens: keyword arguments914 (17.34%), semantic arguments1496 (28.38%), read arguments1388 (26.33%), FINAL1474 (27.96%). Nine exact repeated calls among148 calls. Source hashes and checked pairwise prefill/generation accounting: lab/reports/decision_current_policy_refresh48_20261024/generation_accounting.json. These are opportunities, not demonstrated savings. Removing every read-argument token would still be only an optimistic26.33% ceiling on this training sample.

After current GRPO paired evaluation: (a) if quality remains high but generation is expensive, prioritize a bounded same-checkpoint read-argument selection/reuse pilot, preserving useful search and comparing with always-generate; account for both a matched basic baseline and the fixed historical target. (b) If quality falls with early FINAL, prioritize refreshed current-policy quality-preserving action targets rather than stronger cost penalties. (c) If those targets are reliable but poorly fitted, test the exact-initialization context residual candidate on matched data. Probability confidence may gate selection versus generation, but calibration must be learned/checked on training-only partitions, not assumed from maximum softmax.

No new GPU experiment launched by this accounting. Independent confirmation remains protected; learned policy quality and end-to-end generation engineering must be reported separately.

Train-only refreshed target analysis:57exact-prefix matches show mean absolute action-F1 change13.73pp across suffix policies/seeds (confounded). Fresh nonzero continue-vs-FINAL contrasts agree in29/37two-repeat pairs (78.38%). Prefer repeat-consistent or confidence-weighted targets and reference retention for ambiguous states if another offline update is warranted; do not hard-label the whole cache. Best-action tie overlap is misleadingly high. Full evidence: lab/reports/decision_current_policy_refresh48_20261024/target_comparison.json.

Read-selection feasibility check: existing public-context candidates exactly reproduce20/33trainingreadcalls, corresponding860/5272totalgeneratedtokens (16.31% optimistic ceiling).24/33reads selectoneID. This rules out claiming existing candidate reuse alone achieves30% savings on this sample. Keep generationfallback; a learned expanded pointer/selector would need new training coverage and paired evaluation. Wait for current GRPO results before choosing this versus refreshed policy update.
