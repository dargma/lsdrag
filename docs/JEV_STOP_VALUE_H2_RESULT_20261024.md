# H2 result: continuation improves quality, but fails the saving hypothesis

One frozen-tool stopping preference,129 learned parameters; no architecture
grid. Greedy H1 training collection completed48questions/83states/214episodes.
One fit, followed by128 paired development episodes in196.93seconds,0errors.
H1 reference replay is exact on all64; raw totals and Drive SHA verified.

| Policy | F1 (%) | Generated tokens |
|---|---:|---:|
| Fixed basic GRPO |22.50|98.41|
| H1 |28.30|75.34|
| H2 stopping preference |29.61|79.28|

H2 vs H1: +1.3021F1pp,2improvements/0losses; marginal paired95% CI[0,3.3854]pp.
Generation increases5.23%, ratioCI[.968,1.166]. Against fixed basic GRPO,
quality+7.11pp/generated−19.43%, still fails the30% saving target. Exposed64,
one training seed and adaptive comparisons; no independent goal confirmation.

The two F1 improvements cost172 additional tokens: one partial answer becomes
fully correct by F1, and one zero-F1 answer becomes partially correct. Other
already-correct answers cost66 more tokens in aggregate, while both-zero
answers cost14 more. Net252 additional tokens. Thus this is not simply all
wasted exploration, but the learned quality-first preference does not decide
whether its incremental quality is worth the extra generation.

Keep H1 as the efficient incumbent and preserve H2 as a higher-quality,
higher-cost candidate. The promised cost-saving hypothesis is negative in this
pilot. Tool choices are frozen at an identical state, but stopping changes
subsequent visited states; this is not a claim of unchanged full trajectories.

Next large hypothesis: allocate continuation by its expected quality gain
relative to generation cost, rather than treating every positive quality gain
as a reason to continue. Prefer a single train-only budget-aware candidate using
the verified greedy H1 outcome pairs; do not sweep penalties on dev or globally
increase the stopping bias. Freeze the concrete objective before fitting.
Both-zero cheaper outcomes remain excluded from positive improvement targets.
No additional data/architecture matrix is authorized by this report itself.

Goal ACTIVE; reserved80 untouched. Evidence:
`lab/reports/decision_stop_value_h2_validation64_20261024/`.
Checkpoint1b0a8bf8928d918f7501ee428d70374d0d083e303686931b4c83c21ba8e800db.
