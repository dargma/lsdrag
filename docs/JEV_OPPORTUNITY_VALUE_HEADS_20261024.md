# Stopping opportunity, separated values and head structure

Authorized by the active research goal. Five fixed configurations share377 training states from192 questions; no new LLM collection. All legal actions have two actual downstream suffix samples under frozen original BC. This is offline policy improvement, not GRPO and not current-policy ground truth.

## Fixed comparisons

|Configuration|Body|Stopping loss coefficient|Value auxiliary coefficient|
|---|---|---:|---:|
|mlp_control|Concat MLP|0|0|
|mlp_stop|Concat MLP|1|0|
|mlp_stop_value|Concat MLP|1|0.1|
|attention_stop|Action-query attention|1|0|
|attention_stop_value|Action-query attention|1|0.1|

All inputs are identical slots0/1/4 (last token, question-end, last observation mean). MLP initializes policy/encoder/normalization exactly from the pre-GRPO selected head, checked against all cached logits. Attention uses shared slot projection128 and four learned action queries;100epochs train-only distillation from the same selected head, LR3e-4. Normalization copied/frozen. MLP984716 versus attention330252 total parameters: structure and capacity/warmstart differ, so do not claim a pure capacity-matched architectural comparison. The within-body auxiliary comparison is matched. Inference uses policy logits; auxiliary predictions do not directly rerank actions.

## Objective

Observed action utility U(s,a)=mean(F1)−0.03 mean(remaining cost). All-zero-quality states have utility zero for every action in every configuration; KL/entropy stay active. Base objective is expected utility with KL0.02 and entropy0.01 relative to that body's initial policy. State weights give each question equal aggregate influence.

For each state with any positive observed quality and legal FINAL plus continuation, choose the best observed continuing action a*. Let Δ=U(s,a*)−U(s,FINAL) and d=logit(a*)−logit(FINAL). Stopping loss is |Δ| softplus(−sign(Δ)d). Ties and all-zero-quality states contribute zero. This is pairwise offline supervision, not an unbiased on-policy estimator. Best-of-noisy-action estimates may be optimistic; downstream validation is necessary.

Value auxiliary: per-legal-action squared error for sigmoid F1 prediction plus smooth-L1 error for positive log(1+cost) prediction, averaged across legal actions and question-weighted states. Predictions are not guaranteed calibrated future success/cost probabilities. Training may improve these regressions without improving behavior.

All configurations:100full-batch AdamW epochs, LR1e-4, gradient norm clip1, seed20261056. Fixed final checkpoints only; no epoch/weight sweep. Actual CPU training10.03seconds. Plan and source/input hashes frozen in lab/registry/decision_opportunity_value_20261024/plan.json before fitting. Train metrics in lab/reports/decision_opportunity_value_20261024/training.json are not heldout results.

## Evaluation and order

Five candidates×same exposed validation64 =320episodes. Unchanged shared corpus, model, prompt, tool semantics and generator. Lightweight wrapper adds only checkpoint loading. Cached prior selected/basic-GRPO results remain the comparison references; primary performance target still refers to fixed basic GRPO, not the new mlp_control. Pairwise comparisons against mlp_control diagnose interventions; report all candidates, no fresh-test claims.

Wait for the current zero_gate coordinator to finish its full train/audit/evaluation/archive before starting this GPU evaluation. Do not rely only on a released GPU lock between phases. Queue verifies the predecessor process identity and terminal state; failure triggers inspection, never implicit restart. Evaluation1800second runtime cap/2100second coordinator cap. Derived checkpoints and raw evaluation get verified Drive archives. Final80/reserved80 untouched.

Coordinator: scripts/coordinate_decision_value_evaluation.py
Frozen evaluator spec: lab/registry/decision_opportunity_value_20261024/validation_spec.json
Results: /content/local_fast/lsdrag-text-v1/results/decision_opportunity_value_20261024
Validation: /content/local_fast/lsdrag-text-v1/results/decision_opportunity_value_validation64_20261024

Next decisions depend on actual validation and probability/behavior diagnostics. Poor off-policy transfer calls for bounded current-policy counterfactual refresh, not larger blind fitting sweeps. Goal remains active regardless of this phase's execution completion.
