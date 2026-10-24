# Decision-policy experiment budget

Completed and failed-run totals are checked against raw episodes. Failed costs remain separate in ledger.json and are not discarded. Running values are provisional status snapshots.

|Setting|Run|Status|Episodes|Generated tokens|Input tokens|
|---|---|---|---:|---:|---:|
|historical_grpo_reference|decision_grpo96_20261023|complete|1392|83182|5557999|
|restricted_pool_exploration|decision_state_outcomes_20261023|complete|1428|57425|4342053|
|restricted_pool_exploration|decision_state_validation1_20261023|complete|320|16817|1127816|
|restricted_pool_exploration|decision_state_validation2_20261023|complete|64|3270|214885|
|restricted_pool_exploration|decision_state_validation3_20261023|complete|96|4251|249215|
|restricted_pool_exploration|decision_grpo_continuation_20261023|complete|1216|67189|4448164|
|restricted_pool_exploration|decision_state_outcomes_seed2_20261023|complete|1410|60874|4387681|
|restricted_pool_exploration|decision_state_validation_seed2_20261023|complete|64|3663|244206|
|restricted_pool_exploration|decision_state_policy_iteration_20261023|complete|2876|100079|7411609|
|shared_corpus_restart|decision_shared_calibration16_20261023|complete|48|4207|181838|
|shared_corpus_restart|decision_shared_teacher192_20261023|complete|192|20560|927195|
|shared_corpus_restart|decision_shared_initialization_validation16_20261023|complete|16|1606|76365|
|shared_corpus_restart|decision_shared_grpo192_20261023|complete|1536|156625|7351308|
|shared_corpus_restart|decision_shared_outcomes192_20261023|complete|2824|190746|10484261|
|shared_corpus_restart|decision_shared_validation64_20261023|complete|384|37713|1897481|
|shared_corpus_restart|decision_shared_structure_validation64_20261023|complete|128|11650|564793|
|shared_corpus_restart|decision_shared_test80_20261023|complete|320|28561|1319714|
|selected_head_grpo_continuation|decision_selected_head_grpo192_20261024|failed|768|61608|3416046|
|selected_head_grpo_continuation|decision_selected_head_grpo192_20261024_r1|complete|1536|119793|6566548|
|selected_head_grpo_continuation|decision_selected_head_grpo_validation64_20261024_r1|complete|128|8316|502063|
|adaptive_policy_followup|decision_selected_head_grpo192_20261024_zero_gate|complete|1536|126791|6868988|
|adaptive_policy_followup|decision_selected_head_grpo_validation64_20261024_zero_gate|complete|128|10264|546977|
|adaptive_policy_followup|decision_opportunity_value_validation64_20261024|complete|320|41550|2658076|
|adaptive_policy_followup|decision_constrained_policy_validation64_20261024|complete|192|14580|915340|
|adaptive_policy_followup|decision_current_policy_refresh48_20261024|collection|7|1073|25478|

## Scope

Named decision-policy campaign only; historical PPO/preparatory runs outside this list are excluded. Shared-corpus and restricted-pool results are separate experimental settings.

- Running counts are status snapshots, not certified terminal totals. Pending runs contribute no invented cost.
- CPU cache fitting and corpus embedding/indexing are not LLM generated tokens and are not included in these token totals.
- Wall times may include loading/waiting and shared-GPU effects; not controlled comparative latency.
- Historical GRPO reference training is separate from new exploration cost; no claim every listed run occurred inside original eight-hour window.
