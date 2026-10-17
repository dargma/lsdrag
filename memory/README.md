# Repository memory

This folder is the recovery entry point for future sessions. Git is authoritative; filesystem/runtime state must be checked.

Read order:
1. EXPERIMENT_STATUS.json if present — newest automated phase checkpoint; verify actual artifacts.
2. CURRENT.md — active objective, last verified state, next action.
3. USER_PREFERENCES.md — durable user instructions and working preferences.
4. HEAD_EXPERIMENT.md — exact active experiment protocol and commands.
5. SESSION_LOG.md — cumulative decisions, invalidated work and milestones.

Canonical code/Git: `/content/local_fast/lsdrag-code`, remote `https://github.com/dargma/lsdrag.git`, branch `jev_v1.1`. The user-visible Drive workspace is a backup/mirror, not the compute or Git execution location.

If local fast was reset, clone the remote branch there, then read this folder before restoring model/data inputs. Never infer a job is still running from an old PID alone. Do not resurrect discarded runs. Keep this memory in Git and append rather than overwriting the historical log.

Latest active scope (2026-10-16): text-only integrated action+argument control; `docs/JEV_TEXT_INTEGRATED_EXECUTION_20261016.md` (v5 implicit references; old planning document is historical). Historical multimodal instructions do not authorize resuming old runs. Read current headers/status before any work.

Latest plan override2026-10-17: `docs/JEV_TEXT_INTEGRATED_IMPROVEMENT_V3_20261017.md` merges action/read heads. Plan saved only; no automatic launch.
