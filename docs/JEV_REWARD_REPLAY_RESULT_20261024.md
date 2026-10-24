# Matched total versus remaining token credit

Exposed pairedvalidation64.

|Policy|F1|Generatedtokens|
|---|---:|---:|
|FixedbasicGRPO|22.50|98.41|
|QualityMLPbefore|32.45|139.03|
|HistoricalmixedGRPO|34.75|150.91|
|Actualtoken totalcredit|35.21|142.34|
|Actualtoken remainingcredit|32.92|127.12|

Two newarms sharebatch,initialization,baseline,scale,optimizer,KL/entropyandlengthweights;onlypastcostremovalchanges. Remainingcredit deltaF1-2.29ppCI[-7.75,+2.73]pp;generatedratio.89308CI[.81642,.95911]. Costreductiondoesnotestablishqualitypreservation. Historicalvsnewarms changesreward/normalization/baseline,so notpurecreditcomparison. Baselinehistoricalreproductionmatchedalltensors exactly beforetrainingnewarms. NoLLMrolloutcollection,headfitonly.

Neithernewarmmeetsgeneration<=68.884375;goalunmet. Validationmultipleexposure/singleseed/noindependentconfirmation. Botharchives independentlyrehashedandrawtotalschecked inverification.json.

Nextinvestigation broadens toargumentandreference-IDgeneration costs, whilepreservingpolicyqualityobjective. Anyinterfacechange needsmatchedbaselinesandseparatereportingofengineeringversuslearnedpolicygains. No blindcostweightgrid planned.
