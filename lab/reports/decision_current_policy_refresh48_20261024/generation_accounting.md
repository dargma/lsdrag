# Train48 generation accounting

48 stochastic training collector episodes only; intervention suffixes excluded

| Action | Decisions | Generated tokens | Share |
|---|---:|---:|---:|
| keyword_search | 47 | 914 | 17.34% |
| semantic_search | 68 | 1496 | 28.38% |
| read_chunk | 33 | 1388 | 26.33% |
| FINAL | 48 | 1474 | 27.96% |

{"exact_repeated_tool_calls": 9, "questions": 48, "generated": 5272, "tool_calls": 148, "error_episodes": 0}

Accounting only, not measured savings.
Repeated identical calls can return useful context; removal needs evaluation.
All read generation is an optimistic upper bound for read-argument savings, not an achievable estimate.
No validation or test access; stochastic training sample is not validation performance.
