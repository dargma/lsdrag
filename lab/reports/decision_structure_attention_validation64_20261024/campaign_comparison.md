# Development comparison

|Policy|F1 (%)|Generated tokens|Input tokens|Calls|Zero-tool episodes|Errors|
|---|---:|---:|---:|---:|---:|---:|
|basic_grpo|22.50|98.41|4625.31|2.33|0|0|
|native|24.70|97.44|4691.03|2.38|1|0|
|selected_before|25.18|76.75|3870.83|1.78|18|0|
|selected_ungated_grpo|22.95|53.19|3973.91|1.42|33|0|
|selected_reference|25.18|76.75|3870.83|1.78|18|0|
|control|16.55|50.36|3005.83|1.22|37|0|
|depth|16.55|58.61|3052.81|1.27|36|1|
|input_gate|16.55|58.94|3059.12|1.28|36|1|
|depth_input_gate|16.55|59.64|3112.42|1.30|35|1|
|observation|16.55|58.98|2974.94|1.25|36|1|
|question_attention|18.37|51.48|2756.33|1.11|38|0|
|action_attention|17.72|62.91|3292.66|1.27|37|1|
|action_set|16.55|48.56|2877.30|1.17|38|0|

Paired64,10000 question bootstrap,marginal95%,single-seed,no multiplicity correction. Development threshold is not goal success. Cached references explicitly identified; prior optional state-storage fix and new checkpoint loader are source differences. Final80/reserved80 excluded.

Goal remains active; this report does not establish independent generalization.
