# Same-state offline target refresh, 2×2 comparison

Frozen prospectively while selected48collection is running. Fit MUST wait for complete collection and passed raw audit. Uses old377cache and newd987cache intersection by(qid,prefixSHA256). Require all48fixedtrainqids represented, legalmask/repeatcounts equal. No outcome filtering. UsefreshFP32pack forboth old/fresh arms, identicalquestionweights, referenceprobabilities andoptimizerbudget. Bind completedfreshcachehash and matchedprefixlist BEFOREoptimization. Unmatchednewstates preservedforfuture,notmixedinto thiscontrolledcomparison.

Arms: old-targetMLP, fresh-targetMLP, old-targetfrozen-questionattention, fresh-targetfrozen-questionattention. Startd987; MLPbase trainable,attentionbase frozen; crossarchitecturecontrast includes adaptationregime,notpurearchitectureeffect. Withinarchitecture only targets change. Originald987 fifthreference. Fixedseed20261061,100updates,AdamWlr1e-4,clip1,Q=F1−.03mixedcost,KL.02,entropy.01,lastcheckpointonly. NoRL/PPO/GRPO. No lossfiltering basedoncontaminatedretrospectiveholdout.

Targetsource changes suffixpolicy andrandomseeds; noisy2samples/action. Matchingprefixcontrolsstate/input,doesnotisolatepolicyfromsamplingnoise. Priorhead already learnedoldtargets; baselinepretrainingexposure disclosed. Freshdata outcomes notindependenttest. Allzeroqualitystates retained undertheunchangedobjective forcausalcomparability; inspecttheirfrequency beforedrawingconclusions.

Afterfittingverifyweights/baseparity andfreeze5×64evaluationspec underunchangednativeprotocol. Reservedtestprotected. Plan isimplemented,notexecuted untilcollectioncomplete. Goal remainsACTIVE.
