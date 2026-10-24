# Matched-data context residual pilot

Freeze before fit. One candidate: same105empiricalParetopairs/32freshstates, samefullfresh88/old377KLretention, same100epochs/LR1e-4/KL.1/seed20261060 as ordinaryPareto. No additional sampling, reward tuning or validation-based epoch choice.

Start exact qualityMLP0a1b; freeze entirebase. Width64 question+actionqueries attend32orderedfullcontextbinmeans fromexisting37slotpacks. These are pooledprefixbins, not rawtokens, not future observations. Zeroresidualoutput ensures exactstartingpolicylogits. Normalizebins usingoldtrain192states only. Learnresidualweights only. Base/aux/norm unchanged. Capacity/trainableparameterization differ, so not isolated causaltest ofinputlocation. HistoricalPolicyHead already supportsbinattention;no noveltyclaim.

Five tests passed covering exactinitialization, input sensitivity, gradients/basefreeze/legalmask and conservativepreference construction. Fitter asserts initial logits equal onallfresh/retentionstates, allfinite/frozenbase unchanged. Finalcheckpoint only.

Evaluation: same frozen validation64, corpus101962, Qwen4B, textprompt and runtime; threeconditions quality_start/ordinaryPareto/contextPareto, max1800sec,192episodes. Existingtworesultsets rerun toverifyparity andpermitmatchedcomparison. Newloader only. Reservedtest protected; no worldmodel. Independentconfirmationstillrequired. Failure does not endactivegoal.
