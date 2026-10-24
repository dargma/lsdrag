# Frozen-base adapter result — 2026-10-24

256paired developmentepisodes,320.20sec,16881generated/917532input. Referenceexactparity64 verified. Finalrawtotals andDrivearchive SHA independentlychecked. Baseweights exactlyd987, onlynewpathslearned;noPPOGRPO/nativeprotocolchanges.

| Policy | F1 (%) | Generated | Zero-tool/64 |
|---|---:|---:|---:|
| BasicGRPO |22.50|98.41|0|
| Native |24.70|97.44|1|
| Originalselected |25.18|76.75|18|
| Depth frozen |19.93|55.44|35|
| Questionattention frozen |22.28|55.50|32|
| Actionattention frozen |20.06|76.08|24|

Compared to matching full-head training, F1 point recovery:depth+3.39pp CI[-3.39,+10.94],questionattention+3.91pp CI[-2.08,+10.94],actionattention+2.34pp CI[-3.91,+9.38]. Single-seed marginalpairedbootstrap,10000draws,exposedvalidation; intervalsincludenoimprovement. Frozenquestionattention approachesbasicGRPO qualitywith43.60%fewertokens butdoesnotmeet+5ppF1goal. It loses2.90ppversusoriginalselected. GoalACTIVE,notachieved.

Neitherfreezingbasenoraddingattention hasestablishedqualityimprovement. Oldpolicy2-suffixQ, in-sampleoptimization andstate-distributiondrift areunresolved. Do notclaimoldtargets areproven cause. Do notkeepblindlyvaryingarchitectures/penalties.

Nextboundedpriority: collectofflinebranchoutcomes using selectedd987 itself onfixedtrain-onlysubset, then compareMLP/questionattention onsamefreshdata while preservingoriginalhead. Earlierrefresh48 useddifferentqualityhead0a1b andcannotbe mislabeledd987-currentpolicy. Keepteacher/split/corpus/formatfixed,recordtotalcollectioncost, freezeprotocolbeforeexecution. No testaccess/worldmodel/GRPO. Newcollectionnotyetlaunched.
