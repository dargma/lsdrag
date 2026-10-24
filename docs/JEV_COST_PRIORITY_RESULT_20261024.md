# Cost-priority correction result — 2026-10-24

128paireddev episodes,reference26.41checkpoint reproduced64exactly onanswer/support/F1/tokens/calls/errors. RawcountsandDrivearchive independentlyverified. Originald98725.18/76.75 preserved; nofreshtest.

| Policy | F1 (%) | Generated | Zero-tool/64 |
|---|---:|---:|---:|
| Fixed basicGRPO |22.50|98.41|0|
| Original d987 |25.18|76.75|18|
| Quality correction MLP |26.41|81.55|16|
| Cost-priority MLP |26.74|75.45|19|

Vsqualitycorrection +.33ppCI[-4.35,+5.02],2wins2losses;tokens−7.47%,ratioCI[.824,1.021]. Vsfixedbasic+4.24ppCI[-2.01,+11.16],tokens−23.32%,ratioCI[.651,.877]. Jointgoalstillfails:requiredF1≥27.5007/gen≤68.8844. Marginalbootstrap,singleseed,exposed64,multipleadaptivecomparisons;noindependentconfirmation.

## Mechanism limitation
Fourquerieschangedfromtools tozero-tools,saving403tokens,totalF1change0:allFOURwereF1=0beforeANDafter. Other60net+13tokens andF1sum+.214286. Totalnetsaving390tokens/64=6.09375. Thuscostsavingprimarilycomesfromabandoningpreviouslyunsuccessfulquestions,NOTdemonstratedmoreefficientcorrectanswerpaths. State-conditionedactionchangesexist (includingoneFINAL→semantic),butthisdoesnotprovelearnedfutureinformationvalueorprincipledunsolvabilityprediction. KeepgoalmechanismrequirementOPEN.

Thisisbestjointpointamongcurrentofflineefficientbranch,notglobalbestF1historically. Preservecheckpointhashinspec;donotmarkachieved. Nextprioritizecoverageofquality-improvingcontinue/readchoices onstilluncollectedtrainquestions andverifybenefitonsuccessfultrajectories, ratherthanrewardinguniversaltermination. Remaining96trainquestionsnotyetcoveredbycurrent-policybranchcollection;predeclareboundednextsubset beforecollection. No newrunatthisreport.
