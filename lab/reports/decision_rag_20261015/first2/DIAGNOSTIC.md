# Stage1 diagnostic

train diagnostic; no learned-head/test/latency superiority claim

Complete: True; expected 2 questions × 3 policies.

| Policy | n | EM | F1 | Mean calls | Mean prompt tokens | Mean completion tokens |
|---|---:|---:|---:|---:|---:|---:|
| native | 2 | 0.0000 | 0.3350 | 1.50 | 8165.0 | 229.0 |
| generated | 2 | 0.0000 | 0.0000 | 3.00 | 41766.5 | 221.0 |
| first_token | 2 | 0.0000 | 0.0000 | 3.00 | 41766.5 | 213.0 |

Support-document hits do not prove evidence sufficiency or correct image use.
Native jointly generates tool names/arguments; its generation cost cannot be split.
Observed wall time is uncontrolled; same-call output budget is not equal total compute.
Observations are diagnostics, not established causal failure labels.
