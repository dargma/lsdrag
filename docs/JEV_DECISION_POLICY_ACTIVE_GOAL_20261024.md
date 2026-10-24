# Active goal: attached decision policy, not experiment completion

## Latest authorization
2026-10-24: user explicitly restarted goal mode. Execute the three proposed experiment axes, add head architecture variants, and if the target is still unmet develop and execute further evidence-driven experiments. Save plans/status/results to Git. This supersedes earlier no-automatic-expansion notes for this campaign. World-model/memory expansion remains deferred.

## Success criteria
1. On the same paired evaluation, improve F1 by at least5 percentage points AND reduce generated tokens by at least30% relative to the fixed shared-corpus basic-head GRPO baseline (checkpoint59a0f76672a3411abaa53ab8fd8f40e0c8ddc027d26d566fea25a7187e4421ef).
2. Establish evidence of a useful state-conditioned action policy: choices/probabilities respond to evidence and the downstream benefit/cost of continuing, rather than merely unconditional tool preference or cheap early failure. Architectural action selection alone establishes functionality, not policy quality or novelty.
3. Validation is for development. Freeze candidate and confirmation protocol before independent evaluation; do not claim success from exposed validation alone. Preserve historical final80 result; reserved80 stays protected throughout development. Report paired uncertainty, single-seed limits and all failures. Similar mean F1 with substantially lower generation is encouraging but does not establish noninferiority.

Goal status remains ACTIVE if experiments finish but criteria fail. Never redefine the goal as execution completion to mark achieved. Continue within efficient bounded runs; do not promise eventual success or suppress negative results.

## Adaptive experiment program
| Axis | Intervention | Current status |
|---|---|---|
|1: cheap-failure gradient|All-four-zero-F1 groups get zero policy advantage, KL/entropy retained. Same pre-GRPO selected checkpoint/settings.|Running: zero_gate coordinator330710, learner330716; verified live. Snapshot560/1536 episodes, round0 question140/192.|
|2: stopping opportunity cost|Learn the same-state value difference between FINAL now and search/read followed by answer, from actual train-only downstream outcomes.|Authorized, not implemented/launched. Inspect and reuse existing all-action outcome cache first; collect bounded on-policy branches only if cache mismatch warrants it.|
|3: separated quality and cost|Predict action-conditioned answer quality and remaining cost separately; use these signals to improve the shared policy representation/action choice.|Authorized, not implemented/launched. Compare against scalar-reward policy; do not assume calibrated probabilities or prediction gains imply QA gains.|

Architecture is integrated into axes2/3 rather than a full Cartesian sweep:
- Control: current multi-position concatenation MLP (last token, question-end, last-observation mean).
- CandidateA: small learned attention/gating over available state slots. First keep identical slots and normalization to isolate the aggregation/body change; additional token positions only if evidence warrants.
- CandidateB: shared small encoder with action-specific quality/cost prediction outputs plus policy logits. Compare identical-input current body versus modified body under matched data/training budget. Predicted utility may assist training without adding a separate language model.
- Stage on CPU/train-cache where possible, then bounded closed-loop validation. Freeze configurations before reading their validation outputs; record capacity, trainable parameters, probability normalization, legal-action masks and actual inference overhead.
- Reuse one shared train-only branch dataset across opportunity-cost and quality/cost supervision. Avoid redundant GPU collection and exhaustive ablations. Combine promising changes only after individual effects are interpretable. Keep unsuccessful configurations/results.

## Evidence and reporting
Track paired F1, generated tokens, input tokens, calls, zero-tool/early-FINAL rate, errors, probability/entropy diagnostics, and actual training cost. Use train-only states with contrasting evidence/opportunity for policy diagnostics; confidence is not correctness. No controlled latency claim on shared GPU. Primary numeric comparator remains fixed basic-head GRPO; ungated selected-head GRPO is the local intervention control.

Fixed environment: Qwen3-4B-Instruct-2507 nonthinking/frozen backbone, four text actions keyword_search/semantic_search/read_chunk/FINAL, shared101962-document corpus, train192/validation64. Single GPU; do not terminate unrelated processes. Phase-specific protocol/checkpoint/source hashes must be frozen before launch. No bulky artifacts or model weights in Git; verified Drive archives for derived runtime artifacts.

Current phase protocol: docs/JEV_SELECTED_HEAD_ZERO_GATE_20261024.md.
Current coordinator: /content/local_fast/lsdrag-text-v1/results/decision_selected_head_grpo_coordinator_20261024_zero_gate.json.
Next: finish axis1 and its audit/evaluation/archive; inspect train-only all-action cache for shared axis2/3 supervision while GPU collection runs. This document is campaign authorization/strategy, not a claim axes2/3 are already implemented.

Git: latest supplied session AGENTS instruction requests remote jev_v1.1; local working branch currently jev_v1.2. Save commits locally and fast-forward push HEAD:jev_v1.1, without force. Historical branch conventions are preserved as history.
