# Constrained-policy pilot: cached feasibility did not transfer

Same exposed validation64, no original final80/reserved80 used. Joint target not met.

|Policy|F1 (%)|Generated|Input|Calls|Zero-tool|
|---|---:|---:|---:|---:|---:|
|BasicGRPO|22.50|98.41|4625|2.33|0/64|
|Quality starting head|32.45|139.03|7672|3.41|5/64|
|Cached budget70|24.46|79.31|5825|2.20|30/64|
|Cached budget55|3.39|9.47|805|0|64/64|

Budget70 versus basicGRPO: F1+1.9636pp,paired95%CI[−6.564,+10.513]pp;generation−19.403%,ratioCI[.6102,1.0173]. Targetfalse. Budget55 reduced tokens by90.38% but lost19.1153pp F1 and always chose immediateFINAL. This is failure, not useful efficiency.

Budget70 had met the offline quality floor and token budget; actual rollout quality retention failed. Budget55 already failed the offline quality floor and also collapsed in deployment. Cached state-weighted outcomes under olderBC continuations are not actual trajectory constraints. Off-policy value/state-distribution mismatch is a plausible limitation, not an isolated causal proof; finite-data overfit, policy realization and optimizer behavior also remain possible contributors.

Threeconditions×64=192episodes/291.32seconds/generated14580/input915340. Starting-head64episodes exactly reproduce previous answers/F1/tokens/calls/actions. Checkpoints/evaluation archives rehashed and raw totals verified. Reports lab/reports/decision_constrained_policy_validation64_20261024. No controlled latency or generalization claim.

Next: execute bounded current-policy train48 counterfactual refresh under the quality-leading starting policy, preserving original train192 pool. Protocol JEV_CURRENT_POLICY_REFRESH_20261024.md. Stop repeating stale-cache coefficient sweeps. Fit the next update only after auditing fresh outcome coverage/provenance. Goal remains ACTIVE.
