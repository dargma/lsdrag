# Stage1 diagnostic

train diagnostic; no learned-head/test/latency superiority claim

Complete: True; expected 20 questions × 3 policies.

| Policy | n | EM | F1 | Mean calls | Mean prompt tokens | Mean completion tokens |
|---|---:|---:|---:|---:|---:|---:|
| native | 20 | 0.0500 | 0.1115 | 1.95 | 11271.3 | 261.3 |
| generated | 20 | 0.0500 | 0.1105 | 3.60 | 53233.8 | 254.1 |
| first_token | 20 | 0.0500 | 0.1105 | 3.60 | 53233.8 | 245.1 |

Support-document hits do not prove evidence sufficiency or correct image use.
Native jointly generates tool names/arguments; its generation cost cannot be split.
Observed wall time is uncontrolled; same-call output budget is not equal total compute.
Observations are diagnostics, not established causal failure labels.
