# Constrained quality/generated-token pilot

Adaptive follow-up authorized by the active goal. Earlier value-auxiliary MLP reached development F132.45 but139.03generated tokens, missing the joint target. Start exactly from that checkpoint; frozen backbone and normalization. No new teacher, data expansion or test access.

Plan frozen before fitting: lab/registry/decision_constrained_policy_20261024/plan.json. Shared377states/train192, all-action actual F1 and suffix-generated-token counts from2fixed-BC continuations/action. Same family-disjoint validation64. These are off-policy state-weighted suffix estimates, not whole-episode performance guarantees.

Define Qθ=E_state E_action[F1], Gθ=E_state E_action[generated suffix tokens]. Maximize Qθ subject to Gθ≤B and Qθ≥Qstart−0.02. Two predeclared budgets: B=.70 or.55 times the fixed basicGRPO policy's expected generated count on this same cache. The second condition probes stricter pressure; neither is claimed to equal its actual rollout ratio. No validation-derived labels or per-question rules.

Primal loss: −Qθ + λt(Gθ/B−1) + λq((Qfloor−Qθ)/.05) +.02KL(πθ||πstart)−.005entropy +.1existing value auxiliary. Duals start0 and update projected ascent with step.05/clamp[0,20]. Actual cached generated counts enforce the constraint; the retained auxiliary cost output still predicts log(1+mixed cost), not tokens, and is not used to impose the budget. All head parameters train, normalizers fixed. No stopping ranking loss: previous comparison showed no incremental benefit. The quality floor is an aggregate training condition, not per-question protection, so cheap failure remains possible on individual states.

Seed20261057,300full-batch AdamW epochs,LR1e−4,gradient clip1. Fixed final checkpoints, no epoch selection. Two CPU tests verify dual gradient direction and known constrained LP optimum with illegal actions excluded. Optimistic per-state randomized-policy LP bounds distinguish lack of feasible cache targets from neural/optimizer failure; LP feasibility is not deployment feasibility.

Training completed11.01CPU seconds:
- BasicGRPO cache G=70.8891; starting Q=.451121/G65.5265; floor.431121.
- budget70: B49.6224,Q.458766,G49.5964; both training constraints met. Unrestricted Q upper bound.460652.
- budget55: B38.9890,Q.416688,G37.5552; quality floor failed despite optimistic LP bound.434411. Preserve and evaluate final candidate; no false feasibility claim.

Both candidates plus exact quality starting head evaluated on fixed exposed64,192episodes,1800second runtime cap/2100coordinator cap. Original final80/reserved80 remain protected. Main target still F1+5pp/generated−30% versus fixed shared basicGRPO. Report actual generation/input/calls/errors/early termination and paired uncertainty; do not substitute training constraints for the goal.

Commands: PYTHONPATH=src OPENBLAS_NUM_THREADS=2 /content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/coordinate_decision_constrained_evaluation.py
Runtime: /content/local_fast/lsdrag-text-v1/results/decision_constrained_policy_20261024
Next: if cache constraints fail to transfer, inspect cheap stopping/long read costs and collect bounded current-policy training counterfactuals before further coefficient sweeps. Goal ACTIVE.
