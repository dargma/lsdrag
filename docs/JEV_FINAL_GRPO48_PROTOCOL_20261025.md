# Final authorized GRPO experiment

Latest user `해봐` authorizes the main experiment after the completed smoke. Exactly one fixed run, no additional candidates/coefficients/seeds/rounds after results. Earlier smoke-only hold lifted for this protocol only. Preserve historicalsmoke; do not initialize from smokeupdatedweights.

Start original9B value_aux checkpoint5191c60eee742695031856b2c66a2b3581995b65ffa78b6347cc9a84d9b476d7. Same frozenQwen3.5-9B reader, commonJSON protocol, corpus, prompt,4actions, parser/handlers, greedyargument/finalgenerator. Existing actionMLP net.* trainable; reader and normalization frozen. No critic or LLM token-policy gradient.

Reward unchanged from smoke: R=EM+0.25F1−0.10T/(T+256)+0.02format_valid. T counts all actual generatedarguments/answers, includingknownwrapperboundarytokens; not just finalanswerlength. Format requires existingtoolargument/FINALJSONschemas, completeFINAL/noerrors. Grammarconstrainedformatmayalreadybeconstant. Ifallgroupformatsvalid itsbonus cancels; reportvariation. Accuracycoefficientdominatesrawboundedcostdifference, but groupnormalizationcanamplifycostdifferencesinallwronggroups; this risk is preserved without hiding or retrospectivegating.

Training: all existing fixedoutcome_train48,4freshon-policy trajectories each=192. Onecollectionunderinitialpolicy followedbyoneupdatephase,2epochs/minibatch16AdamW1e-5,clipratio.2,KL.02toinitialpolicy,noentropybonus,gradientclip1,seed20261085. Same settingsassmokeexceptsize. All4actions sampledfromactualpolicy,epsilon0; FP32 packs andoldlogps stored. ForcedFINALexcludedfromactorloss butallgeneratedtokensincludedinreward. Question/trajectorybalanced loss. No oldcounterfactualbranches asGRPObatch. No epochcheckpointselection. Savefinalhead only.

Trainingceilings7200sec,1milliongenerated,50millioninput. Require40GiBfree beforeGPUload; sharedGPUallowed, unrelatedjobsuntouched. Sequentialcoordinator owns execution andhalts onerror, noautomaticretry. Preservecompletedrawrecords onfailure; donotrestartfromfirststageblindly.

AfterCPUofficialscore/reward/format/token/advantage/behaviorreplay(checkatol/rtol2e-4)/checkpointaudit, archiveallraws+FP32batch+headtoDrive20261025 withhashverification. Then paireddevelopment64×before/after=128episodes, max3600sec. Freezeevaluationspec withactualtrainedcheckpointhashbeforelaunch. Requirebefore64exactreplayofpreviousvalue_auxanswers/support/F1/tokens/calls/errors/actionorder. Noreserved80/priorfinal80. Repeateddev64, notindependentconfirmation. No controlledlatencyclaim.

Reportwhole-setF1/EM/input/generated/toolcalls/errors; correct/incorrect/partialF1costs andsamequestionbothcorrect/gained/lost/bothwrongtotals/means. Prespecifieddescriptivescreen:F1andEMatleaststartingpolicy AND strictgeneratedreductiononbothEMcorrect ANDzeroexecutionerrors. Wrong-onlyshorteningdoesnotpass. Reportfailurewithoutretuning. Verifyrawhashes,backups,finalmemory/Gitpush thenSTOP.

Registry `lab/registry/qwen35_final_grpo48_20261025/`; run `/content/local_fast/lsdrag-text-v1/results/qwen35_final_grpo48_20261025`; evaluatorrun `qwen35_final_grpo_validation64_20261025`. Coordinator `scripts/continue_qwen35_final_grpo48.py`; status `results/qwen35_final_grpo48_20261025_pipeline/status.json`. Audit scripts separatelysaved. Sourcesfrozen beforelaunch.
