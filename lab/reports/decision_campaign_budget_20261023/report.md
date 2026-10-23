# Decision-policy experiment budget

Completed-run totals are checked against raw episodes. Running values are provisional status snapshots.

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
|shared_corpus_restart|decision_shared_grpo192_20261023|collection|480|49895|2269430|
|shared_corpus_restart|decision_shared_outcomes192_20261023|not_started|—|—|—|
|shared_corpus_restart|decision_shared_validation64_20261023|not_started|—|—|—|
|shared_corpus_restart|decision_shared_structure_validation64_20261023|not_started|—|—|—|
|shared_corpus_restart|decision_shared_test160_20261023|not_started|—|—|—|

## Scope

Named decision-policy campaign only; historical PPO/preparatory runs outside this list are excluded. Shared-corpus and restricted-pool results are separate experimental settings.

- Running counts are status snapshots, not certified terminal totals. Pending runs contribute no invented cost.
- CPU cache fitting and corpus embedding/indexing are not LLM generated tokens and are not included in these token totals.
- Wall times may include loading/waiting and shared-GPU effects; not controlled comparative latency.
- Historical GRPO reference training is separate from new exploration cost; no claim every listed run occurred inside original eight-hour window.
