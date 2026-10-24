# Refresh downstream outcomes under the current quality-leading policy

Reason: the constrained70 head satisfied cached quality/token conditions but actual validation fell from F132.45 to24.46; constrained55 collapsed to FINAL for64/64. Old fixed-BC state/continuation outcomes do not provide an actual quality guarantee for the updated policy. Do not retune many coefficients against the same stale cache.

Collect under frozen mlp_stop_value checkpoint0a1b3547cb22f7df5bc781a8a5b4aaa39843d3d046a4b2a28a2ef7f0df966021. Backbone/model/corpus/action interface unchanged. Keep original train192 pool and validation64 split. Choose exactly48training questions by SHA256('refresh48_20261058:'+qid) ordering, fixed before collection; no validation outcomes used to choose questions.

For each question sample a current-policy collector trajectory; reconstruct initial and last nonforced decision state (deduplicate if only one). At each state execute every legal action with2sampled continuations under this same frozen policy. Save actual F1, generated/input/call costs, raw arguments/tools/answers and prefix hashes. These are intervention outcomes for offline improvement, not on-policy GRPO samples. No independent judge. FP32packs, legal-action/prefix/hidden equality checks. Sampling seed20261058 with deterministic per-question/turn/rep offsets. All failures retained.

Bound:48questions,atmost96states/816episodes,3600seconds after model load,coordinator3900seconds,1Mgenerated/60Minput caps. Shared-GPU serialized lock. Only collection/audit/archive automatically chained; the next fitting protocol will be frozen after inspecting train-only coverage and discrepancy. Original final80/reserved80 untouched. No world-model expansion; empty transitions artifact retained only for historical schema compatibility.

Coordinator scripts/coordinate_decision_current_refresh.py runs collection, verifies raw episode totals/all-action sample completeness/current-policy prefix identity and archives exact bytes. Runtime /content/local_fast/lsdrag-text-v1/results/decision_current_policy_refresh48_20261024.

Next: compare overlapping initial-state action outcomes with old-BC suffix targets to quantify policy mismatch; inspect learnable stopping/retrieval opportunities. Fit bounded policy update using explicit outcome-source provenance and question weights. Do not silently mix old/new targets or claim48questions replace train192. Goal remains ACTIVE.
