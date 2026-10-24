# H3 result: token threshold passes, quality and useful-saving claim do not

One train-only budget-target stopping head (936e4592), no parameter sweep.
Paired development64 completed128episodes/183.92seconds,9103generated,
453629input,0errors. H1reference replay exact64; raw report checks and full
Drive archive bytes verified. This is exposed development, not confirmation.

| Policy | F1 (%) | Generated tokens |
|---|---:|---:|
| Fixed basic GRPO |22.50|98.41|
| H1 |28.30|75.34|
| H2 |29.61|79.28|
| H3 budget stopping |27.32|66.89|

H3 crosses token threshold68.8844 but misses F1threshold27.5007.
Against H1,3questions improve and4worsen; F1−.9884pp, generated−11.22%.
Against fixed baseline,+4.8152pp and generated−32.03%; joint goal NOT met.
Do not tune on the small numerical miss or round it into success.

Of541net tokens saved vsH1,305come from40both-zero questions and143from
4quality-loss questions.3improvements save33tokens and17positive-retained
questions save60. Useful savings exist but do not dominate. This does not
establish the intended useful quality-preserving calculation policy.

Training feasibility improved after filling missing intermediate states:
83initial/last +54middle=137prefixes on48train questions;15changedtargets
in11questions. Targets include3quality gains,10ties and2quality losses under
an aggregate quality floor. Such training compensation is not guaranteed to
transfer to new questions; the development losses are consistent with this
limitation, not proof of a unique cause. Only129parameters were trained;
backbone, H1base and three tool logits remain frozen at the same input.

Collection failure preserved: inherited snapshot helper produced34rather than
54middle states; independent coverage check rejected it. Supplement exactly
20missing states, no repeated rollin/heldout usage.108new suffix episodes,
5200generated/319427input/0errors. Runtime durations69.91+48.13seconds;
merged verification counts.seconds is inherited first-run time, not total.
Both archives are verified. Fit then ran once with frozen100updates/.001.

Keep H1 efficient incumbent; H3 is an experimental cost-focused branch, not a
replacement certified against the joint goal. Next direction: broaden evidence
for quality-preserving intermediate stopping under the same H1policy, rather
than compensate one question's lost quality with another's gain. Start with a
bounded training-only coverage expansion and one candidate; no dev penalty
sweep or architecture matrix. Exact followup split/protocol not yet frozen,
no followup worker launched. Preserve H2 higher-quality/cost branch.

GoalACTIVE; reserved80 untouched. Reports:
lab/reports/decision_budget_stop_h3_validation64_20261024/.
