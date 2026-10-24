# Matched reward replay execution

Saved design and arithmetic tests complete; autonomous goal continuation now prepares one matched replay experiment. Existing768onpolicyepisodes used only fromexactbehavior0a1b. Historical mixedreward/control must reproduce finalc6b0checkpoint within1e-6before fitting two new candidates. GPUhead-only4epochs360updates/LR1e-4, same permutationseed20261059,clip.2,KL.02,entropy.01,inversetrajectorystepweights. NoLLMrolloutcollection; no newpolicyiteration.

Two new arms use coefficient.0003onactualgeneratedtokens, leave-one-out episodebaseline, commontrain-wide rewardstd.467204...and existingallzeroF1gate. token_total repeatsfullreward; token_remaining uses reward-to-go. They differ only in pastcostremoval. Comparison tohistorical alsochangesreward/normalization/baseline andmustnotbeascribedtocredit alone. Keep auxiliaryoutputweights/norm/backbone frozen. Forcedtailscostincluded;forcedactionsnottrained.

Fivearithmetic tests passed. Exactcheckpoint/input/sourcehashes frozen before fit. Recheckrawepisodehashes, oldlogprob2e-4,allfinite,fourchangedtensors. Ifreproductionfails stop andretainfailure. Finalcheckpoints only; atmosttwonewcandidates onunchangedvalidation64,1200sec eachcombinedcap1800; cachedquality_start/GRPO references with explicitprovenance. Noindependenttestaccess. Noqualityorjointsuccessclaimfromfit.
