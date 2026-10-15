# Current state — 2026-10-15 UTC

Active objective: execute the full six-action head-only pilot step by step after a brief Astra protocol check. No world model or surprise component in this run.

Code/Git: `/content/local_fast/lsdrag-code` on `jev_v1.1`. New files awaiting the next milestone commit are the full-head collector/trainer/evaluator, its tests, the application brainstorm, protocol and this memory. Model, data and execution stay on local fast; Drive only receives backups.

Verified:
- Full-tool direct A-RAG implementation and prior Astra audit are committed; actual-backbone synthetic functional smoke passed all five tools, both images and FINAL. This was not a trained policy benchmark.
- Original final288 inputs restored:288 documents,56822 chunks,2438 figures. All pixel SHA/PIL checks and Gemma composite index hashes passed.
- New outcome-blind split:100 ordered train questions,20 development; train prefixes60/80/100. Dev has0 overlap with the historical100 questions;51 train questions overlap those historical inputs. No old predictions/labels loaded. Support+public-table source groups do not cross new splits; global corpus remains shared, so no unseen-source claim.
- Full-head pilot source passed5 targeted CPU tests; combined direct-head tests passed27. Astra approved the protocol and requested safeguards for coverage, shared prefix and infrastructure failure handling; those safeguards were implemented.

Current execution: first teacher attempt started after Astra approval and commit b99a264. It was stopped after observing native tool-call output where the controlled selector required A-F. This attempt is INVALID_PROTOCOL_ATTEMPT.json, not head quality evidence and not reusable training labels. The owned teacher PID777960 was stopped; tool session31865 is historical, not a live-job assertion.

Next: v11_backend agent is testing a latest-user A-F readout instruction with the full tool schema on two fixed train questions using the real GPU model. All three policies must use identical selection preprocessing; the new feature contract must bind that preprocessing. After this format-only correction passes and Astra checks it, start a new output identity on the same fixed question order. No outcome-based question selection or gate changes. Do not resume the invalid first attempt. Then complete60/80/100 and coverage-gated training/evaluation as planned.

Do not resume `/content/local_fast/lsdrag-v11/results/stage12_20261015_v2`: it has DISCARDED.json. Old runners also reject discarded outputs.
