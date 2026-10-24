# Empirical quality-preserving preference result

Exposed validation64, frozen shared corpus/reader.

|Policy|F1|Generated tokens|
|---|---:|---:|
|FixedbasicGRPO|22.50|98.41|
|QualityMLP|32.45|139.03|
|Refreshed empirical Pareto|28.17|104.80|

Token generation fell24.62% versus qualityMLP, but F1 fell4.28pp (paired95%CI[-10.40,+1.43]pp). The training criterion did not establish quality retention in evaluation. Against fixedbasicGRPO, F1 pointgain exceeds5pp but tokens are higher, so the joint target fails. No independent confirmation or claim of guaranteed dominance.

Next: one matched-data context-residual architecture candidate, keeping pair targets and reference retention fixed. This asks whether broader context access helps distinguish when a cheaper action is appropriate. Frozen base and zero-initialized residual preserve initial logits; existing32pooledbins, not rawtokenattention. Not a pureequalparametercomparison. No new reward sweep.

Evidence: lab/reports/decision_pareto_refresh_validation64_20261024/report.md and campaign_comparison.json. Goal remains active.
