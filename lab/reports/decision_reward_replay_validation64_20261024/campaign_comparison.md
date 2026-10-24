# Development comparison

|Policy|F1 (%)|Generated tokens|Input tokens|Calls|Zero-tool episodes|Errors|
|---|---:|---:|---:|---:|---:|---:|
|basic_grpo|22.50|98.41|4625.31|2.33|0|0|
|native|24.70|97.44|4691.03|2.38|1|0|
|selected_before|25.18|76.75|3870.83|1.78|18|0|
|selected_ungated_grpo|22.95|53.19|3973.91|1.42|33|0|
|token_total|35.21|142.34|9967.67|4.11|6|1|
|token_remaining|32.92|127.12|8749.20|3.77|7|1|

Paired64,10000 question bootstrap,marginal95%,single-seed,no multiplicity correction. Development threshold is not goal success. Cached references explicitly identified; prior optional state-storage fix and new checkpoint loader are source differences. Final80/reserved80 excluded.

Goal remains active; this report does not establish independent generalization.
