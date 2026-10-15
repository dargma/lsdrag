# Session recovery and working rules

Read `memory/README.md`, `memory/EXPERIMENT_STATUS.json` if present, `memory/CURRENT.md`, `memory/USER_PREFERENCES.md`, and the latest entries in `memory/SESSION_LOG.md` before changing code or starting/resuming experiments. Read `memory/HEAD_EXPERIMENT.md` for the active protocol and commands. These are repository memory, not proof an experiment completed: verify actual artifacts and processes.

- Work and manage Git in `/content/local_fast/lsdrag-code`. Use local-fast models/data directly. Obtain public inputs from the web into local fast; Drive is for backup. Restore unique derived artifacts from Drive only when exact bytes are unavailable publicly.
- Push code, protocol, memory, reports and small provenance records to branch `jev_v1.1`. Do not commit model weights, credentials, private tokens or bulky raw datasets.
- Current authorization is the head-only full-tool A-RAG pilot. World models/surprise are future research, not part of this experiment. The prior reduced Q3 run was discarded; never resume it or reuse its labels/results.
- Preserve vanilla A-RAG keyword/semantic/read behavior and G1 pixel/text fusion plus actual-image delivery. The new profile has five tools plus FINAL; prompt, schemas, action order, checkpoint and handlers share a contract.
- Shared-GPU functional/quality work is authorized if VRAM permits. Comparative latency measurement is deferred; do not stop merely because another GPU process exists or terminate unrelated jobs.
- Keep memory cumulative: update CURRENT and HEAD_EXPERIMENT at each phase transition; append decisions and failures to SESSION_LOG. Record concrete paths, process/session identifiers when active, hashes, next commands and blockers. Commit/push memory with meaningful milestones. Avoid secrets.
- No stale success claims: distinguish implementation smoke, teacher collection, coverage, learned-policy quality and research novelty. Ask for clarification only when genuinely necessary; continue already-authorized work.
