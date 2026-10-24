# Development comparison

|Policy|F1 (%)|Generated tokens|Input tokens|Calls|Zero-tool episodes|Errors|
|---|---:|---:|---:|---:|---:|---:|
|basic_grpo|22.50|98.41|4625.31|2.33|0|0|
|native|24.70|97.44|4691.03|2.38|1|0|
|selected_before|25.18|76.75|3870.83|1.78|18|0|
|selected_ungated_grpo|22.95|53.19|3973.91|1.42|33|0|
|selected_reference|28.30|75.34|3670.03|1.64|20|0|
|stop_value|29.61|79.28|4001.86|1.83|16|0|

Paired64,10000 question bootstrap,marginal95%,single-seed,no multiplicity correction. Development threshold is not goal success. Cached references explicitly identified; prior optional state-storage fix and new checkpoint loader are source differences. Final80/reserved80 excluded.

Goal remains active; this report does not establish independent generalization.
