# Train-only structure policy diagnostic

No held-out or causal claim. No candidate filtering. Action order: keyword, semantic, read, FINAL.

| Policy | Common train loss | Matched Q | Q above shuffled | Entropy |
|---|---:|---:|---:|---:|
| selected_reference | -0.00498 | 0.36804 | 0.15314 | 0.4978 |
| control | -0.03467 | 0.40496 | 0.20142 | 0.2990 |
| depth | -0.03754 | 0.40909 | 0.20649 | 0.2497 |
| input_gate | -0.03519 | 0.40589 | 0.20291 | 0.2825 |
| depth_input_gate | -0.03774 | 0.40948 | 0.20707 | 0.2423 |
| observation | -0.03713 | 0.40841 | 0.20563 | 0.2594 |
| question_attention | -0.03708 | 0.40810 | 0.20499 | 0.2769 |
| action_attention | -0.03843 | 0.41028 | 0.20779 | 0.2432 |
| action_set | -0.03604 | 0.40675 | 0.20342 | 0.2849 |
