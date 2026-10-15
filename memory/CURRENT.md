# Current state — 2026-10-15 UTC

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
