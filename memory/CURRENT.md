# Latest recovery override — 2026-10-15 UTC

Canonical Git clone restored from remote branch `jev_v1.1`, previous HEAD `c080bb9`, to `/content/local_fast/lsdrag-code`. The Drive code mirror was older.

User confirms the last discussion concerned minimizing generated tool arguments. User now requires reconsidering and agreeing the design, then updating the plan/code before proceeding. **Do not resume collection, training, evaluation, or the old coordinator.** Model/data restoration is authorized and ongoing. Preserve all five tools plus FINAL, G1 pixel/text fusion and actual image delivery; world models remain deferred.

Active user goal: establish the revised experiment plan and complete through stage2. Stage definitions must be explicit in the revised plan; do not assume the old first60 stage is authorized by this new goal.

Design status: no revised design is agreed or implemented. Candidate for review: select observed IDs via typed candidate slots; reuse existing queries when sufficient; generate a new query only when needed. Check candidate restrictions, quality loss and added control cost before acceptance. `docs/JEV_DIRECT_DISPATCH_HEAD_DESIGN.md` is an older research proposal referring partly to discarded Q3 work, not the recovered final answer or an approved execution plan. The post-stop Astra review outcome has not been recovered.

Verified backup: `/content/drive/MyDrive/lsdrag-backups/20261015/full_head_20261015/v2_teacher_020.tar.gz` contains 20 teacher JSON/PT pairs. Git records 33 completed at the user stop; the extra 13 records have not been located or recovered. Never report 33 recovered. No experiment GPU process was present at initial recovery inspection.

Input recovery active: tool session64295 runs `scripts/restore_full_head_inputs.py --report /content/local_fast/lsdrag-head-v12/recovery --workers 12`; plan2463 files/1060223873 bytes, including2438 referenced figures. Tool session31833 downloads official pinned Gemma E4B and EmbeddingGemma2 snapshots to `/content/local_fast/lsdrag-v11/hf`. Neither operation has been verified complete at this checkpoint. These session IDs are ephemeral: inspect actual processes/files after any reset.

Next: finish/verify input and model restoration; regenerate fixed question inputs via `scripts/prepare_full_head_inputs.py`; restore pinned scorer; record integrity results. Separately finish argument-control design review and obtain agreement before modifying the execution protocol or running experiments.

Historical snapshots below are retained for audit and overridden by this entry.

# Latest override — 2026-10-15

User explicitly stopped the running pilot for Astra review and action/argument-control replanning. Coordinator792503, collector796306 and descendant796517 are confirmed exited. 33 completed teacher records preserved in full_head_20261015_v2; see STOPPED_BY_USER.json. Do not auto-resume. Earlier running-process descriptions below are historical.

# Current state — 2026-10-15 UTC

Read `memory/EXPERIMENT_STATUS.json` if present: the coordinator updates and publishes it at each newer phase boundary. The narrative below is a dated handoff snapshot, not a claim the listed phase is still live.

Active objective: execute the full six-action head-only pilot step by step after a brief Astra protocol check. No world model or surprise component in this run.

Code/Git: `/content/local_fast/lsdrag-code` on `jev_v1.1`. Code, protocol, application brainstorm and memory were committed/pushed at b99a264 and ad47382. Model, data and execution stay on local fast; Drive only receives backups.

Verified:
- Full-tool direct A-RAG implementation and prior Astra audit are committed; actual-backbone synthetic functional smoke passed all five tools, both images and FINAL. This was not a trained policy benchmark.
- Original final288 inputs restored:288 documents,56822 chunks,2438 figures. All pixel SHA/PIL checks and Gemma composite index hashes passed.
- New outcome-blind split:100 ordered train questions,20 development; train prefixes60/80/100. Dev has0 overlap with the historical100 questions;51 train questions overlap those historical inputs. No old predictions/labels loaded. Support+public-table source groups do not cross new splits; global corpus remains shared, so no unseen-source claim.
- Full-head pilot source passed5 targeted CPU tests; combined direct-head tests passed27. Astra approved the protocol and requested safeguards for coverage, shared prefix and infrastructure failure handling; those safeguards were implemented.

Current execution: first teacher attempt started after Astra approval and commit b99a264. It was stopped after observing native tool-call output where the controlled selector required A-F. This attempt is INVALID_PROTOCOL_ATTEMPT.json, not head quality evidence and not reusable training labels. The owned teacher PID777960 was stopped; tool session31865 is historical, not a live-job assertion.

Current live phase: v2 first20 collection resumed from12 valid saved question artifacts after an unknown SIGTERM terminated the initial collector and coordinator. No Python exception/infrastructure failure marker was recorded. Valid12 artifacts remain; incomplete question13 is recomputed under unchanged source/data identity.

Detached collector PID789249, coordinator PID789250 (no live tool-session ID). Read run/launch.json and verify actual commandlines before acting. Logs: `/content/local_fast/lsdrag-head-v12/logs/teacher_v2_000_020_resume.log` and `/content/local_fast/lsdrag-head-v12/logs/coordinator_v2_resume.log`. The initial tool sessions95665/77816 are dead and must not be polled as live jobs.

Next: inspect completed20-question diagnostic, archive phase artifacts, then continue unchanged train order to60. Extend80/100 only as the fixed coverage gate requires. Fit/evaluate only when six-action coverage passes. Source/data identities are frozen; do not edit active runner/core files mid-run. World-model work remains deferred.


Do not resume `/content/local_fast/lsdrag-v11/results/stage12_20261015_v2`: it has DISCARDED.json. Old runners also reject discarded outputs.

The detached coordinator in launch.json waits for the resumed collector before advancing fixed stages. See HEAD_EXPERIMENT.md and run/coordinator_status.json. Do not launch parallel GPU collectors for this run.
