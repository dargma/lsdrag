# Development comparison

|Policy|F1 (%)|Generated tokens|Input tokens|Calls|Zero-tool episodes|Errors|
|---|---:|---:|---:|---:|---:|---:|
|basic_grpo|22.50|98.41|4625.31|2.33|0|0|
|native|24.70|97.44|4691.03|2.38|1|0|
|selected_before|25.18|76.75|3870.83|1.78|18|0|
|selected_ungated_grpo|22.95|53.19|3973.91|1.42|33|0|
|mlp_control|29.14|114.69|6255.47|2.86|7|1|
|mlp_stop|29.09|114.80|6377.39|2.92|7|1|
|mlp_stop_value|32.45|139.03|7671.84|3.41|5|2|
|attention_stop|30.21|136.92|10263.41|3.78|4|1|
|attention_stop_value|30.21|143.78|10964.33|3.94|4|1|

Paired64,10000 question bootstrap,marginal95%,single-seed,no multiplicity correction. Development threshold is not goal success. Cached references explicitly identified; prior optional state-storage fix and new checkpoint loader are source differences. Final80/reserved80 excluded.

Goal remains active; this report does not establish independent generalization.
