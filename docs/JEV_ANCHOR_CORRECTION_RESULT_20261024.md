# Local correction result — 2026-10-24

192pairedvalidationepisodes,289.78sec,15124generated/751910input. Referenceexact64parity;rawtotals andDrivearchive independentlyverified. Noerrors in3conditions. Noreservedtest.

| Policy | F1 (%) | Generated | Zero-tool/64 |
|---|---:|---:|---:|
| Fixed basic GRPO |22.50|98.41|0|
| Original selected |25.18|76.75|18|
| Local correction MLP |26.41|81.55|16|
| Local correction frozen question attention |25.18|78.02|19|

MLPvsoriginal+1.23F1pp,pairedCI[-1.56,+5.13],2wins/1loss;tokens+6.25%CIratio[.975,1.185]. Againstfixedbasic+3.91pp,CI[-3.91,+12.5],tokens−17.13%CIratio[.707,.943]. Joint+5pp/−30%notmet. Smallimprovementnotstatisticallyestablished. Attentionper-questionF1identicalreferenceandnoefficiencygain. Amongthisoffline-efficientstructurebranch,MLPisnowpoint-qualityleader; historicalhigh-quality/high-costheads stillhavehigherF1.

Selectivequalitycorrection avoidslargequalitycollapseobservedwithglobalofflineutilityoptimization,butthisisa smalladaptiveexposeddevresult. Source58/82states andobjectives differfromolderruns;no architecture-onlyclaim. Fewerzero-toolanswers plausiblycontributequalitygain,butneedactualtrajectoryevidencebeforemechanismclaim.

Nextfocusquality-preservingcostcorrections ratherthananotherheadarchitecture. Currentcorrectionselection prioritizesqualitygain,allowingmoreexpensiveactions;thiswaspredeclaredand explainswhyjointcostimprovementwasnotguaranteed. Inspecttrainonly9empiricalnonworse-cheaperpairs, relativeactionsofnewMLP andoriginalanchor, beforefreezingasinglecost-awaretarget variant. Avoidblindlambda/epochgrids; preservesameprotocol/retentioncontrol. Newexperimentnotlaunchedatthisreport. GoalACTIVE.
