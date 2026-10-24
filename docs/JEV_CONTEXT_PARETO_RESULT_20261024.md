# Context-residual result

Paired exposed validation64, threeconditions.

|Policy|F1|Generated|Calls|
|---|---:|---:|---:|
|QualityMLP|32.45|139.03|3.41|
|Ordinary empiricalPareto|28.17|104.80|2.53|
|Contextresidual empiricalPareto|26.51|102.14|2.69|

Contextcandidate does not improve the quality/cost tradeoff over ordinaryPareto in thispilot. Both frozenreferenceconditions reproducedall64answers/support/F1/generated/input/calls/errors exactly. VersusqualityMLP, F1delta-5.94pp with marginal95%CI[-11.65,-1.26],generationratio.73466. FixedbasicGRPO22.50/98.41 remains jointtarget reference: neither required F1gain nor30%generationreduction met. Noindependenttestaccess.

192episodes/430.65sec,raw22142generated/1246289input verified. Botharchives rehashed independently. Evidence: lab/reports/decision_context_pareto_validation64_20261024/verification.json, report.md, campaign_comparison.json. Broadercontextaccess andfrozenbaseparameterization change together; this does not rejectallattentionarchitectures.

Reward/creditassignment design saved peruserrequest in JEV_ARAG_HEAD_REWARD_DESIGN_20261024.md. No newrewardrun launched. Goal remains active.
