# Full six-action head pilot: Astra execution review

Date: 2026-10-15. Decision: **approved to start the fixed first-20-question teacher diagnostic and continue under the frozen gates. No remaining execution blocker was identified in the reviewed implementation.** This approves a behavior-cloning feasibility experiment; it does not certify a trained head's quality.

The scope matches the requested head-only experiment. The six alternatives are five retrieval/reading tools plus FINAL, chosen iteratively. There is no world model, surprise objective, UPR, or reuse of the discarded three-action predictions or labels. The frozen plan is [execution_plan.json](../lab/registry/full_head_20261015/execution_plan.json).

Reviewed `src/lsdrag/head_experiment.py`, `src/lsdrag/direct_reader.py`, `scripts/run_full_head_pilot.py`, and `tests/test_head_experiment.py`. Generated A–F selection, constrained first-token selection, and the learned head use the same decision prefix and common downstream argument/answer generation. Teacher features are copied before action decoding, avoiding label contamination. Forced FINAL decisions are excluded; natural decisions from unsuccessful episodes remain eligible. Train-only feature normalization is correctly folded into the linear checkpoint.

The identified blockers are resolved: training requires a complete, exact fixed prefix of 60, 80, or 100 questions, at least 30 distinct teacher-label questions, and five distinct questions for each action. Artifact identity and feature contracts are checked. Infrastructure failures, including tool-worker/index failures stored inside trajectories, stop collection rather than becoming apparent controller failures. Invalid controller outputs remain scored failures. Checkpoint contracts prevent silent reuse of incompatible heads. The focused CPU suite passed **5/5** in this review; the root additionally reports 27 combined CPU tests passing after migration to `/content/local_fast/lsdrag-code`.

Interpretation and continuation conditions:

- The first 20 questions diagnose functionality and teacher behavior; they cannot authorize fitting. If coverage still fails at the fixed 100-question maximum, report NoGo or a clearly identified partial-action result without claiming six-action learning. Do not force labels or select questions by observed outcomes.
- This is imitation of a generated selector, not optimal tool selection. Sparse image-search/figure-read use, premature FINAL, and poor teacher answers can limit learned quality even if coverage passes.
- Twenty held-out development questions support descriptive pilot comparisons, not a final-test or noninferiority claim. The manifest records zero support/table overlap between selected train and development groups and zero development overlap with the prior 100 questions; 51 training questions were previously human-seen. The shared corpus and prior exposure must remain disclosed, without claiming strict unseen-source evaluation.
- The short-letter/common-arguments baseline is not native A-RAG. Native A-RAG, if evaluated, is a separate reference with its serialization differences disclosed. Report all three fixed head seeds rather than choosing the best development result.
- Shared-GPU execution permits functional and quality observations, not comparative latency claims. The earlier functional smoke used a scripted synthetic head; this pilot still needs real teacher collection, fitting, and held-out rollout evidence before any learned-quality claim.

The initial diagnostic should also confirm valid A–F decoding at the actual generation boundary and preserve selector/argument token accounting. This is a diagnostic check, not a change to the frozen selection or training protocol.
