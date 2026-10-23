# Shared-corpus decision-policy restart

2026-10-23 user override: question-specific candidate pools may make semantic-search→answer too easy; restarting experiments is authorized. This protocol supersedes the remaining small-pool training/evaluation queue. The 22:48 UTC budget remains; no world-model or memory expansion.

## Verified issue and claim boundary

All 256 current training/validation questions have 20 paragraphs, including two supporting paragraphs. Search returns up to the generated top_k (default five), title plus the first 300 characters; median paragraph length is 395.5 characters. Runtime allowlists excluded gold labels. This is a restricted candidate-pool setting, not demonstrated label leakage, and not adequate by itself to establish open-corpus adaptive retrieval. Selected seed1 greedily used semantic_search→FINAL for every original validation32 question. Existing results remain historical; no retrospective rewriting or discarding failures.

Stop coordinator127863 before its next fit/evaluation. Collector129475 may finish and be archived, but it must not trigger further small-pool training. Prepared fixed-rule validation queue was never launched and is superseded. Existing final test160 remains unscored.

## Corpus and tools

Build one shared corpus from **all** public paragraph text/title fields in the locally available official MuSiQue train/dev files, exact title+text deduplication and content-hash IDs. Every question searches exactly the same documents. No per-question 20-paragraph filter, question-specific gold injection, support-based distractor selection, answer field, or decomposition in runtime. Test-source public documents may be in the retrieval corpus, but test questions/labels are never training examples. This is dataset-derived shared-corpus retrieval, not full Wikipedia or an official full-corpus benchmark.

Use a reusable dense embedding index and lexical index on local fast. Preserve keyword_search, semantic_search, read_chunk, FINAL and the nonthinking Qwen3-4B generator. Synchronize corpus IDs, legal actions, schema, prompt, read handler and references. Keep search snippets at 300 characters initially so corpus changes are not confounded with deliberately hiding evidence. Do not force reading or multiple calls merely to make the head appear useful. Freeze top_k limits and search observation budget for all compared policies.

## Calibration before policy learning

Use only already-exposed validation questions for calibration, chosen deterministically before results; never final test160. First measure original-question retrieval support Recall@5 and all-support coverage, then compare semantic-once→FINAL with the original GRPO and a native generated-action policy on a bounded common set. Record F1, retrieved evidence coverage, actual read evidence, generated/input tokens, errors and query counts. Answer text occurring in a snippet alone does not establish reasoning success or leakage.

No arbitrary universal minimum pool size is claimed. Prefer the complete deduplicated pool over choosing whichever sampled size makes our method win. If multi-step methods cannot access supporting evidence, diagnose retrieval/chunking; if semantic-once succeeds, count it as a valid strong baseline. We need both easy and harder questions, not a hand-selected set requiring many calls. Do not select difficulty or corpus based on our head's advantage. Additional nested-pool sweeps are optional diagnostics, not the main benchmark.

## Learning and evaluation restart

Freeze corpus/index/prompt/tool hashes, training192/validation64/final160 question and family splits, seeds and budgets. Existing cached hidden states, branch outcomes and small-pool checkpoints are not valid new-environment training data. Recollect trajectories and same-state action outcomes in the shared environment. A prior head may be an explicitly labeled initialization/control, not proof of retraining or fair from-scratch superiority. Establish a competent common initialization using train-only native trajectories or train-only teacher distillation before comparing GRPO and probability-weighted action outcomes with matched initial checkpoints and recorded budgets.

Reuse verified learning implementations and the limited input/body candidates; do not rerun every old matrix. Prioritize one simple head, one promising multi-position head only if validation justifies it, plus semantic-once and GRPO controls. Fresh heldout comparison is run only after candidate selection is frozen. Evaluate all methods on identical corpus, retriever, generator and token accounting. Report prior small-pool results separately. The target remains F1 +5 percentage points / generated tokens −30%, now against a GRPO reference evaluated in the **same shared-corpus environment**; old absolute F1/token values cannot define success here.

Within the remaining time, finish corpus/index, calibration and as much bounded retraining/evaluation as feasible. Do not weaken the evaluation or claim completion merely to fit the deadline. Record incomplete requirements explicitly if necessary. Conditional predictive auxiliary remains deferred unless the genuine primary heldout target is reached under the active protocol.

## Execution resumption after original window

2026-10-23 23:29 UTC clock verification: original22:48UTC deadline had elapsed during the pause/interruption. User subsequently explicitly said “계속 해야지” and goal status returned active. Calibration16 is a newly resumed bounded run (max1800seconds/48episodes), not work completed within the original8-hour window. Further stages require bounded budgets recorded before launch; no unlimited overnight execution inferred. Main learning/evaluation objective remains incomplete.

## Frozen shared-corpus initialization

Collect one greedy native-action trajectory for each of frozen train192, max3600seconds/900k generated/45M input. Keep all non-forced states, including failed-answer episodes, and train only from native action IDs (never dummy uniform probabilities). No previous head weights. Common PolicyHead last-token/MLP128+linear skip, train-only feature normalization/std floor.1, question-balanced hard-action CE with .05 legal-only smoothing, seed20261051,100 full-batch epochs AdamW3e-4, gradclip1. Critic disabled; no validation-driven epoch selection. This is initialization, not the proposed outcome policy. Subsequent GRPO and same-state action-outcome heads start from identical saved weights and same actor trainability; costs tracked separately. Training budget changes are recorded before launch, not inferred as unlimited.
