# Repository memory

This folder is the recovery entry point for future sessions. Git is authoritative; filesystem/runtime state must be checked.

Read order:
1. EXPERIMENT_STATUS.json if present — newest automated phase checkpoint; verify actual artifacts.
2. CURRENT.md — active objective, last verified state, next action.
3. USER_PREFERENCES.md — durable user instructions and working preferences.
4. HEAD_EXPERIMENT.md — exact active experiment protocol and commands.
5. SESSION_LOG.md — cumulative decisions, invalidated work and milestones.

Canonical code/Git: `/content/local_fast/lsdrag-code`, remote `https://github.com/dargma/lsdrag.git`, branch `jev_v1.2`. The user-visible Drive workspace is a backup/mirror, not the compute or Git execution location.

If local fast was reset, clone the remote branch there, then read this folder before restoring model/data inputs. Never infer a job is still running from an old PID alone. Do not resurrect discarded runs. Keep this memory in Git and append rather than overwriting the historical log.

Latest active scope (2026-10-16): text-only integrated action+argument control; `docs/JEV_TEXT_INTEGRATED_EXECUTION_20261016.md` (v5 implicit references; old planning document is historical). Historical multimodal instructions do not authorize resuming old runs. Read current headers/status before any work.

Latest plan override2026-10-17: `docs/JEV_TEXT_INTEGRATED_IMPROVEMENT_V3_20261017.md` merges action/read heads. Plan saved only; no automatic launch.

Latest2026-10-17: Astra direction review complete; CURRENT/HEAD newest headers apply. Proposed small counterfactual probe is not yet implemented/frozen; no new GPU run. Generalization100/30 is deferred by review recommendation, not completed.

Latest2026-10-17: bounded contraction probe completed, no learned compression training; intermediate transition gate failed. Read newest CURRENT/HEAD and contraction_probe_20261017 report. No active worker or automatic next experiment.

Latest2026-10-17: frozen new100 paired evaluation complete; report JEV_FROZEN_HEAD_GENERALIZATION100_REPORT_20261017.md. USER HOLD: save/report/push only, no new experiments until explicit resume. Read newest CURRENT/HEAD.

Latest2026-10-17: Astra deep research complete; all-head online outcome/cost policy improvement recommended, one-intervention flagship rejected. Research proposal only; experiment HOLD remains. See newest CURRENT/HEAD and JEV_HEAD_RESEARCH_DEEP_DIVE_ASTRA_20261017.md.

Latest 2026-10-23: active work/push branch is `jev_v1.2`, per explicit user override.

Latest 2026-10-23: authorized decision policy pilot1 completed on jev_v1.2. PPO quality/step improvement not established. Read newest CURRENT/HEAD and JEV_DECISION_POLICY_PILOT1_REPORT_20261023.md. No active worker or automatic expansion.

Latest2026-10-24: shared-corpus final80x4 complete; numeric target not met. Read CURRENT/HEAD top and docs/JEV_DECISION_SHARED_FINAL_REPORT_20261024.md. No active worker or automatic expansion.
