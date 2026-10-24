# Development comparison

|Policy|F1 (%)|Generated tokens|Input tokens|Calls|Zero-tool episodes|Errors|
|---|---:|---:|---:|---:|---:|---:|
|basic_grpo|22.50|98.41|4625.31|2.33|0|0|
|native|24.70|97.44|4691.03|2.38|1|0|
|selected_before|25.18|76.75|3870.83|1.78|18|0|
|selected_ungated_grpo|22.95|53.19|3973.91|1.42|33|0|
|selected_reference|25.18|76.75|3870.83|1.78|18|0|
|old_mlp|15.96|61.41|3862.91|1.72|34|1|
|old_question_attention|21.29|61.03|3154.70|1.28|28|0|
|fresh_mlp|23.36|77.66|4603.48|2.02|17|0|
|fresh_question_attention|21.44|64.86|3726.88|1.78|27|0|
|fresh_direct_utility|12.02|75.64|4541.50|1.83|26|2|

Paired64,10000 question bootstrap,marginal95%,single-seed,no multiplicity correction. Development threshold is not goal success. Cached references explicitly identified; prior optional state-storage fix and new checkpoint loader are source differences. Final80/reserved80 excluded.

Goal remains active; this report does not establish independent generalization.
