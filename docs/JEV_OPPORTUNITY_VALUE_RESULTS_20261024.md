# Opportunity/value/structure comparison: accuracy gain with increased cost

All five fixed candidates completed on exposed development64. Joint target remains unmet. Training cache/source/architecture distinctions are specified in JEV_OPPORTUNITY_VALUE_HEADS_20261024.md. No original final80 or reserved80 used.

|Policy|F1 (%)|Generated tokens|Input tokens|Calls|Zero-tool|Errors|
|---|---:|---:|---:|---:|---:|---:|
|Fixed basic GRPO|22.50|98.41|4625|2.33|0|0|
|MLP control, further cached policy improvement|29.14|114.69|6255|2.86|7|1|
|MLP + stopping supervision|29.09|114.80|6377|2.92|7|1|
|MLP + stopping + value auxiliary|32.45|139.03|7672|3.41|5|2|
|Attention + stopping|30.21|136.92|10263|3.78|4|1|
|Attention + stopping + value auxiliary|30.21|143.78|10964|3.94|4|1|

Highest F1 candidate (MLP stop+value) versus basicGRPO: +9.9502pp, paired marginal95%CI [+0.2404,+20.1065]pp; generation +41.2829%,ratioCI[1.1643,1.7337]. This is exposed validation, one training seed and five candidates without multiplicity correction; do not claim independent or robust significant generalization. Quality margin is promising for further cost reduction but not proof such reduction is achievable.

Matched comparisons:
- Stopping supervision alone versus MLP control: −0.0558pp,2wins/2losses,gen+0.0954%; no useful incremental effect shown.
- MLP value auxiliary versus same-body stopping: +3.3654pp,3wins/0losses,CI[0,+8.0529]pp;gen+21.1107%. Tradeoff, not a free improvement.
- Attention value auxiliary versus attention stopping: identical per-question F1,gen+5.0097%. No quality gain in this run.
- Attention comparison changes capacity and initialization (distillation), so no general conclusion against attention architectures. Here it offers no preferred accuracy/cost tradeoff.

Training:5fixed checkpoints,10.03CPU seconds on reused train-only outcomes;no new LLM rollouts. Evaluation320episodes,803.87seconds,generated41550/input2658076. All failures retained. Training/eval archives independently rehashed and episode/token totals reconciled. Full means/paired intervals/provenance in lab/reports/decision_opportunity_value_validation64_20261024/campaign_comparison.json. Conditional same-question train-state response probe is not a success criterion and does not replace this evaluation.

## Next adaptive decision

The originally approved three axes have now been evaluated; none meets the joint target. Continue the active goal. Do not simply increase stopping supervision or expand attention sweeps: current evidence does not support those directions.

Use the quality-leading mlp_stop_value checkpoint as a development starting point, not a final selected winner. Next implement a small constrained policy-learning pilot on existing training outcomes: optimize expected answer quality while separately enforcing a generated-token budget, preserving quality with a stated training-only floor. Use actual per-action suffix generated counts rather than the old mixed tool/input/generation cost as the token constraint. Keep the policy small and frozen LLM. Track constraint feasibility and dual variables, not just a mixed scalar reward. Fix at most two training budget levels before evaluation, report all; do not tune them on partial validation.

First check feasibility and fit on CPU cache without new GPU collection. Cached suffixes remain off-policy under originalBC; if apparent training feasibility fails to transfer, collect a bounded current-policy counterfactual refresh on existing training questions before further optimization, not a broad blind coefficient sweep. Freeze the exact constraint/floor definitions and commands before launch. No fresh-test evaluation until an actual candidate/protocol is frozen. Numeric and decision-quality goals remain ACTIVE.
