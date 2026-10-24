# Same-state offline target refresh, 2×2 comparison

Frozen prospectively while selected48collection is running. Fit MUST wait for complete collection and passed raw audit. Uses old377cache and newd987cache intersection by(qid,prefixSHA256). Require all48fixedtrainqids represented, legalmask/repeatcounts equal. No outcome filtering. UsefreshFP32pack forboth old/fresh arms, identicalquestionweights, referenceprobabilities andoptimizerbudget. Bind completedfreshcachehash and matchedprefixlist BEFOREoptimization. Unmatchednewstates preservedforfuture,notmixedinto thiscontrolledcomparison.

Arms: old-targetMLP, fresh-targetMLP, old-targetfrozen-questionattention, fresh-targetfrozen-questionattention. Startd987; MLPbase trainable,attentionbase frozen; crossarchitecturecontrast includes adaptationregime,notpurearchitectureeffect. Withinarchitecture only targets change. Originald987 fifthreference. Fixedseed20261061,100updates,AdamWlr1e-4,clip1,Q=F1−.03mixedcost,KL.02,entropy.01,lastcheckpointonly. NoRL/PPO/GRPO. No lossfiltering basedoncontaminatedretrospectiveholdout.

Targetsource changes suffixpolicy andrandomseeds; noisy2samples/action. Matchingprefixcontrolsstate/input,doesnotisolatepolicyfromsamplingnoise. Priorhead already learnedoldtargets; baselinepretrainingexposure disclosed. Freshdata outcomes notindependenttest. Allzeroqualitystates retained undertheunchangedobjective forcausalcomparability; inspecttheirfrequency beforedrawingconclusions.

Afterfittingverifyweights/baseparity andfreeze5×64evaluationspec underunchangednativeprotocol. Reservedtestprotected. Plan isimplemented,notexecuted untilcollectioncomplete. Goal remainsACTIVE.

## Sequential continuation
Coordinator waits for existing selected-refresh coordinator to completeaudit/archive; verifieslivePID whilewaiting andneverrestartscollector. Thenfit(max600sec),archivefit,evaluate5×64(max3000sec;1Mgenerated/40Minput),reportandarchive. Anyfailedgate haltschain. Validationtemplate is priorunchangednative64spec; candidatecheckpointidentities boundafterfit beforeevaluation. Noindependenttestorautomaticnextcampaign.

## User-approved added direct utility head (before any fitting)
User: “너의 제안도 추가해서 같이 평가실험해”. Addfresh_direct_utility ontheSAMEfreshmatchedprefixrecords. Slots0/1/4→d987initialized128encoder→concatenate16-dimensionallearnedactionembedding→shared64MLP→sigmoidexpectedfractionalF1 andsoftplusremainingmixedcost. Policylogits=(quality−.03cost)/.1,legalmask. Costisexistingmixedcost (calls+generated/100+inputs/10000), notpuregeneratedtokens; no silentlychangedreward. ExpectedF1notcorrectnessprobability; softmaxnotcalibratedsuccessprobability. No newtools/generalizationclaim.

Supervisedloss: qualityMSE + Huber(cost/trainmeancost) + .02KLtod987, legalactionaverage/questionbalanced. Costscaletrainonly;500updates,AdamWlr.001,clip1,seed20261061. Encoderandscorertrainable/backbonefrozen; normalizationfixed. Noinitialidentityclaim; thisarmchangesarchitectureANDobjective. Onecandidateonly/nohyperparametersweep; compareactualF1/generatedtokens. Functionaltestchecksutilitydirectlydetermineslogits,legalmask,andgradients.

Originalwaitingcoordinator389583 stoppedwhilewaiting,collector388356untouched. Supersededwaiterrecordpreserved;replacementusesr1statefile. Revisedchain6×64afterauditedcollection,unchanged3000sec/1Mgenerated/40Minputbudget. Trainingcodehashesupdatedbeforeanyfit. Other4armsunchanged.
