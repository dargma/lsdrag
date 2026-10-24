# Reward/credit matched replay preparation — saved, not launched

User requested ARAG/head-aware ideas saved first. This is a concrete low-cost follow-up proposal, not a running training experiment. No reward optimizer or GPU evaluation launched by preparing this plan.

## Reusable evidence

All768 original qualityMLP on-policy episodes (train192×4) have two reader calls per decision: zero-generation prefill and argument/answer generation. Their2925decisions reconstruct79144generatedtokens exactly.2851decisions are controllable; forcedFINAL contributes2198tokens and must remain in earlier return-to-go even though its action likelihood is excluded. All4replicas/question present; no branch interventions. Evidence: lab/reports/decision_quality_head_grpo192_20261024/reward_replay_feasibility.json.

Restart exact behavior head0a1b, not its updatedGRPOc6b0 descendant. FP32cachedpacks and oldlogprobs permit matching the same update budget across objective variants. Recheck strict replay before updates, validate hashes and question splits. Reuse is legitimate for a matched first policy-update experiment; it is not new current-policy rollout data for later iterations. Full repeated online learning will need fresh collection.

## Sequential experiment definition

A. Reproduce original mixed-cost, episode-group advantage update on identicalbatch/order/LR/epochs/referenceKL/entropy/rowweights, verifying the historical checkpoint or explained numerical tolerance before claiming a matched comparison. No extra LLMrollout cost. Abort interpretation if reproduction fails.

B. Change only episode reward to F1−0.0003*T_generated. This preserves the original marginal generated-token penalty while removing tool/input penalties. Same groupstandardization, allzeroF1gate and everythingelse. This tests objective alignment, not a tuned optimum or budget guarantee.

C. Use same terminal quality and generated-token coefficient, but remaining-generation credit at step t. Let G_i,t=sum_{u>=t}g_i,u and R_i=F1_i−0.0003*G_i,0. Use a question-level baseline from the OTHER3trajectories, b_i=mean_{j!=i}R_j; fixed per-question scale S estimated once from original total rewards with explicit epsilon. Compare total-return and return-to-go versions using THE SAME leave-one-out baseline and scale. To isolate credit rigorously, B must also use this baseline in the matched B-vsC comparison; if original B uses ordinarygroupcentering, add only a CPUupdate control for the baseline change, not a newdatacollection. Do not silently call multiple simultaneous changes 'step credit only'.

A_i,t=(F1_i−0.0003*G_i,t−b_i)/S.

Allzeroquality groups retain the existing gate for this pilot. Forcing/invalidaction policy staysfixed; alltailtokens count. No new critic, auxiliary shaping or dualparameter in the first comparison. Rawreturn-to-go removes pastcost dependence but does not estimate full actioncounterfactualvalue; it is a limited first step. Questiongroup normalization is a practical GRPO-style surrogate, not an unbiased-gradient guarantee.

## Trajectory weighting caveat

Current actor loss averages withintrajectory using1/controllablesteps weights. This prevents longtrajectory rowcountdomination but differs from the sum of decision logprobgradients for an expected episode-return objective. With policy-dependent length, this can change the optimizedsurrogate. Do NOT claim inverse-length weighting solvescreditassignment or is exacttrajectorypolicygradient. Keep it fixed for initial creditcomparison; study sum-based objective separately only if warranted, holding overall gradientscale/updatebudget explicit. This avoids mixing reward,baseline,creditandlengthweighting changes in one uninterpretable result.

## Resources and decision rule

Proposed cap: same original4epochs/minibatch32/LR1e-4, actor/encoder only, allfixednorm/auxweights/LLM. Training cache reused, no LLMcollection. Exactcandidate list and formulas must be finalized before any fit; at most2new evaluationcandidates after CPU reproduction/control checks, same validation64 and fixedbaseline. Priorquality_start and originalGRPO results may be cached references afterparity/source verification; do not repeatedlyspend evaluationcost solelyto reproduceallconditions.

Required tests before execution: futurecost includesforcedFINAL; changingpastcost cannotchangeunscaledremainingcost; noillegal/forcedactions getactorupdates; zerogate preserved; pairedsamebatcholdlogprobs; actualgenerateddefinition matchesrawrecords; fixedparameterchecks; noeval/testlabels. Saveeveryfailure/candidate, finalcheckpointonly, nointermediatevalidationselection.

Interpretation: improvement must be actualpairedF1andgeneratedtokens, not trainingreward. Existingvalidation exposure and singleseed uncertainty remain. Finalnumericgoal requires independentconfirmation and state-conditioneddecision evidence. No newrun is currently authorized by this document alone; existing user sessionauthorization governs later execution, while latest save-first request is respected for this preparation.
