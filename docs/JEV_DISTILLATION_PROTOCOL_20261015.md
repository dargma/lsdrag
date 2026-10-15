# Native teacher → student action head: revised protocol

This protocol supersedes automatic stage2 continuation in JEV_DECISION_RAG_EXECUTION_20261015.md. Existing v13 runs are preserved, not invalidated or relabeled. No old Q3 labels are used. The user authorized code/protocol refinement and automatic experiment start after 20 minutes of GPU idleness. The first automatic experiment is a same-backbone target-format audit, not a larger-reader result or head-quality claim.

## Scientific question and target contract

Output syntax need not match. The target is a semantic action from the five-tool registry plus FINAL, chosen at the same observed state. Compare the existing generated-letter teacher baseline with a native single-action teacher. The latter is explicitly instructed to return one call or a final answer; it is NOT unchanged native A-RAG, which may emit multiple calls.

Student features are extracted with the exact student selection prompt, backbone and image inputs BEFORE teacher generation. The teacher gets the same non-system history, question and images, with its own native instruction. No current teacher action, rationale, answer, arguments, gold or future observation enters the student input. Previous executed actions/arguments are legitimate history and remain intact. Teacher/student system prompts intentionally differ and have separate pinned contracts. We keep the student's existing selector prompt and tool schemas in this first audit; prompt simplification is a later isolated ablation.

The student uses a zero-decode first-token probe to capture its pre-output feature and provide a same-state baseline. It does not supply the teacher label. Once per collection process, compare its captured normalized feature numerically with the direct backbone last hidden state used by a learned head; failure halts collection. Charge this extra check in the recorded costs.

Teacher output must finish normally and contain exactly one known, schema-valid native call or valid final-answer JSON. Reject malformed JSON, truncation, multiple calls, unknown tools and invalid arguments before dispatch and target publication. Never take the first of multiple calls, map parse failure to FINAL, or force missing action coverage. Store raw rejection/teacher output in the failed episode. A valid empty answers list is a natural FINAL, not proof of sufficient evidence. Forced budget FINAL is never an action target.

Teacher call arguments drive teacher rollouts. Student deployment retains its own common argument/answer generator. This creates a teacher-rollout vs student-rollout distribution difference and must be assessed through closed-loop QA/recall, not hidden behind imitation accuracy. Valid format does not establish action quality. Include valid earlier decisions from unsuccessful episodes; separately report execution failures, repeated actions and premature stopping. No outcome labels or RL are introduced.

## Ordered experiments

1. Preserve v13 budget256 (60 episodes) and unfinished common1024 (41 episodes) as their original baselines. Their source is saved at Git c98cccc and the verified user-stop archive. Changed source hashes prohibit resuming those directories using the new source. Use the pinned old source if completing the old diagnostic later.
2. New v14 first2 native-target collection. Check pre-output feature equality, accepted/rejected targets, actual tool execution/image delivery, answer formatting and same-state native-vs-first-token disagreement. Any target rejection, reader error or infrastructure/readout failure stops automatic expansion and produces a review checkpoint. Low F1 alone does not remove questions or stop expansion.
3. If first2 has accepted targets and no such errors, extend unchanged to fixed first20 train questions. Report every question, accepted-state denominators, missing/forced states, action coverage, per-modality recall, costs and answer scores. This audit stops at20; it does not automatically begin60-question collection or claim that labels are good because they parse.
4. Review teacher quality and model-role separation. The user requested a larger reader. Default interpretation pending their optional clarification: final-answer reader only, student decision backbone remains E4B. Gemma-4-12B-it is the next listed size, but has a unified architecture; test its processor/image delivery/native format independently before using it. The current runtime reports transformers5.18.0 and a gemma4_unified config mapping, which is not proof that this repository's HFReader hooks and loader work with it. No larger-model quality or execution claim yet. Teacher enlargement is a separate factor.
5. After the native20 review, freeze the chosen teacher/student/reader contracts and collect fresh train60, extend80/100 only for coverage (>=30 distinct natural-state questions and >=5 distinct questions per each of6 actions). Train linear6 with existing3 fixed seeds and plain CE. Native and letter labels must not be mixed. New backbone/prompt means fresh student features; a matching hidden width is insufficient provenance.
6. Paired dev20: generated-letter, first-token and all3 learned heads share student argument/final path. Native is a reference with a different serialization/rollout policy. Existing evaluator's agreement statistic is against generated-letter heldout states, NOT native-teacher imitation accuracy; a separate same-state native heldout audit is required before making native-imitation claims. Report all seeds and closed-loop behavior, not best seed.
7. Later isolate decision-input simplification and schema-constrained argument decoding. Do not combine those changes with model scaling or silently change frozen runs. No automatic repair/retry that hides format failures.

## Metrics and limitations

- Answer EM/F1, answer-format validity and manual audits of semantically correct wording mismatches. Recall complements F1; it does not fix answer normalization.
- Text/table/image supporting-source recall, with modality-specific eligible denominators. Record first-search source Recall@1/3/5/10 separately from whole-trajectory accumulated source recall. These are document proxies, not exact supporting text spans/cells/images.
- Explicit reads and actual image delivery separately. Search snippets may already supply text/table evidence; lack of a read call is not absence of evidence.
- Reviewed same-visual-content image candidate/open/delivery recall using existing alignment reviews. Exclude differing accepted-figure sets pending reconciliation, report mapped coverage/unresolved IDs, and never extrapolate the mapped subset to all image questions.
- Exact text span/table-cell recall and joint all-required-evidence sufficiency remain unavailable until validated alignments exist. Missing alignment is unknown, not zero or success. Evaluate reader quality on sufficient-evidence subsets only after verification.
- Native/first-token same-state action disagreement is not itself a quality ranking. Report teacher rollout cost, student probe/check cost, and deployed student cost separately. Shared-GPU timings are observations, not comparative latency evidence.

## Execution and automatic idle gate

Common args: `lab/registry/native_distillation_20261015/common_args.json`.
Run: `/content/local_fast/lsdrag-head-v14/results/native_distillation_20261015`.
Pilot: `PYTHONPATH=src python3 -u scripts/run_native_distillation_pilot.py --common-args lab/registry/native_distillation_20261015/common_args.json`.

The persistent local watcher polls the selected GPU UUID every30 seconds. Both GPU compute and memory utilization must be <=5%, with >=32GiB free, for1200 seconds of consecutive observations. Busy samples, unavailable telemetry and observation gaps >65 seconds reset the interval. No prior idle duration is assumed. Recheck telemetry, source hashes and runtime package versions immediately before launching. The worker checks free VRAM again. This is a sampled idle condition, not a GPU reservation; no unrelated process is stopped.

One-shot launch intent and file locks prevent duplicate launches. A changed source/package blocks launch rather than executing a different experiment. Do not automatically retry a failed run. The monitor can survive chat end while this machine remains alive; it cannot survive host/runtime shutdown. STOP file at the recorded watcher state directory cancels waiting and terminates only its own experiment process group. Old v13 STOP records remain historical; this is a newly authorized v14 run.

Each completed/review/failed phase bundles runtime artifacts and plans to Drive, verifies SHA256, and records small reports/memory in Git. If another conversation has tracked edits in progress, defer the Git checkpoint rather than committing those edits; keep the verified Drive checkpoint and a local deferred record. No weights or bulky data enter Git.

Model reference: https://huggingface.co/google/gemma-4-12B-it (Google model card, inspected2026-10-15).
