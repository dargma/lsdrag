# Zero-quality group gate: completed development result

The gate changed stopping behavior but did not meet the joint target. All results below are the same exposed validation64; prior baselines are cached references. No original final80/reserved80 was used.

|Policy|F1 (%)|Generated tokens|Input tokens|Calls|Zero-tool episodes|
|---|---:|---:|---:|---:|---:|
|Basic shared GRPO|22.50|98.41|4625.31|2.33|0|
|Selected head before GRPO|25.18|76.75|3870.83|1.78|18|
|Selected head + ordinary GRPO|22.95|53.19|3973.91|1.42|33|
|Selected head + zero-quality gate GRPO|25.96|83.63|4675.69|2.02|15|

Against ordinary selected-head GRPO, gated F1 rises3.0066pp (paired95%CI −2.983,+9.635pp), generated tokens rise57.23%. Against basic shared GRPO, F1 rises3.4598pp (CI−4.464,+12.054pp), generated tokens fall15.0206% (ratioCI .73544,.96394). Against pre-GRPO selected head, F1 rises0.78125pp and generation rises8.9577%. Target +5pp/−30% not met, even as development point estimates. No quality significance, noninferiority or fresh-test success claim.

Only intervention: set policy advantage to zero for questions where all4 sampled answers have F1=0. Other settings and pre-GRPO starting checkpoint identical. First768 rollout episodes exactly match ungated collection;82groups gated,68 formerly had cost-only advantage variation. All192groups/round independently checked. Same-seed controlled comparison supports an effect of this objective change on stopping behavior; it does not establish general understanding or robust accuracy improvement. KL/entropy remain active on gated groups, so this is not equivalent to dropping those states entirely.

Training1536episodes,4719decisions,4698controllable states,588updates,2409.89seconds. Generated126791/input6868988. Preserved2errors/21forcedFINAL. Final checkpoint dccbf686944009a72ee43251def7139c0ef676e8c84ccc7b50c290ed611fd61f. Both GPU replay errors≤9.54e−7 under unchanged2e−4checks; normalizers fixed and all4MLP tensors changed. No LLM training.

Validation128episodes,202.11seconds,generated10264/input546977,0errors. All64 starting-head episodes exactly match prior results, including answer/support/F1/tokens/calls/actions. Training and evaluation archives independently rehashed; completion_audit.json records identities. Raw results and paired intervals are in lab/reports/decision_selected_head_grpo_validation64_20261024_zero_gate.

Interpretation: the head is trainable as an action policy and responds to return construction; cheap-failure preference is a plausible contributor, with the intervention reducing immediate termination. We still need state-appropriate selective continuation rather than uniformly more search. The fixed opportunity-cost/value/attention candidates are already trained and queued for next evaluation. Their specification was frozen before this result; do not retroactively retune it. Research goal remains ACTIVE.
