# Same-prefix target refresh + direct utility result

384paireddev episodes,527.68sec,26710generated/1520659input. Originalselectedreference64answers/support/F1/tokens/calls/errors exactlyreproduced. Rawtotals/Drivearchive independentlyverified. Noheldouttestaccess. All5trainedcandidates usedsame58matchedprefixstates spanning48trainquestions; backbonefrozen.

| Policy | F1 (%) | Generated | Zero-tool/64 |
|---|---:|---:|---:|
| Basic GRPO |22.50|98.41|0|
| Original selected |25.18|76.75|18|
| Old target MLP |15.96|61.41|34|
| Fresh target MLP |23.36|77.66|17|
| Old target frozen question attention |21.29|61.03|28|
| Fresh target frozen question attention |21.44|64.86|27|
| Fresh direct quality-cost |12.02|75.64|26|

Fresh-minus-old MLP+7.39F1pp CI[.23,15.34],tokens+26.46%. Attention+.16ppCI[-9.11,9.53]. Sameprefixfeatures/actionmasks controlstateinputs; targets differinsuffixpolicy ANDrandomsampling. Thusfresh-targetbenefitforMLP isobservedbutcannotattributeallto policyalone. Neitherbeatsoriginalselectedorjointgoal. Morecurrenttargets reducedearly-stopbias forMLP(34→17zero-tool),no guaranteeheadqualitygeneralizes.

Directutility isclearnegative:versusselected -13.16ppCI[-22.28,-4.71],tokensroughlyunchanged. Train58qualityMAE.0463,mixedcostMAE.0828,utilityMAE.0475,empiricaltie-awareoptimalchoice81.0%,d987agreement74.1%,weightedcachedregret.0357. Thesearein-sample. Reasonabletrainfit doesnot implyqualityheldoutorclosedloop; 500updateson58statesanddirectargmaxofnoisypredictionscanoverfit. No claimwhichcausalmechanismdominates. Recommendationwasahypothesisandfailedthispilot.

Allintervalsmarginalbootstrap,exposed64,singleseed,multiplecandidates. Noindependentconfirmation; goalACTIVE. Keeporiginald987incumbent forcurrentofflineefficientstructurebranch; historicalhighF1/highcostqualityheads remainpreserved.

Next: beforeanotherarchitectureexperiment, inspectfresh82trainstates versusmatched58, including24newlaterstatesandrepeatconsistentadvantageoverd987. Considerboundedconservativecorrection onfreshcoverage, with explicitqualityretention andcostseparation; do notrepeatblindfullpolicyoptimization orclaimpredictionqualityfromtrainingloss. Freezeexactnextprotocolaftertrain-onlydiagnosis. No newrunlaunchedatthisreport.
