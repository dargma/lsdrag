# Reference serialization cost: train-only evidence and next pilot

Train48 stochastic quality-policy collectors generated5272tokens. Existing observed chunkIDs are24-character contenthash strings. Re-encoding only observed chunk_ids/support_ids with short d0,d1 aliases changes JSON tokenization by1763tokens (33.44% of actualgenerated): read1038,FINAL725. Searchtextandanswertext unchanged in accounting.53readreferences/37knownFINALreferences;2unobservedFINALreferences retained unchanged. This is a hypotheticalpayloadencodingdelta, not actualinferencegain, not policyimprovement. Sourcehashes and analysis script saved.

Broader implication: actionpolicy is paying tospell internal references. Removing usefulretrieval is not the only costlever. Shortreferenceencoding could compose with action/readselection heads, but cannot be attributed to learneddecisionquality. At127.12tokens, a33.4%reduction would still be about84.6 (illustrative, not a prediction), abovefixed68.88target. More thanencodingalone maybenecessary.

Proposed boundedpilot: bijective episode-local aliases assigned on first publicobservation; use them consistently in model-visible histories, readschemas, returnedresults and FINALschema, mapback tocanonicalIDs forhandlers/scoring/audit. Stableorder,deduplicateIDs, preserve originalretrievalscores/tieordering anddocuments; do not relabel the storedcorpus or changeembeddingsearch. The model must not gain goldsupports, answers, question-specificretrievalpool or hiddenpriorinformation. Handle unknownFINALreferences explicitly andretainfailures, never silently repair to gold.

Engineering requirements beforelaunch: roundtrip/bijection/legal-reference tests; sameinnerretrievalresults; consistentdispatch/schema/parser/outputmapping; count everygeneratedtokenincludingreferences; retain originalprompt/runtimeprofile. Newinterfacechanges hiddenstates,so existingheads mayneed retraining; don'tclaim exactpolicyparity frommappingcorrectness.

Compare native/basicGRPO and chosenhead under BOTHcanonical andshortreferenceinterfaces wherebudgetpermits; baselinegetsidenticalencodingbenefit. Report(a)fixedhistoricaltarget,(b)matchedinterfaceheadadvantage,(c)encodingonlygain separately. Ifonly(a)passes, itdoesnotprovelearneddecisioncontribution. Use train-onlysmallfunctionalpilotfirst, freezeprotocol/candidatebeforevalidation64; protectreservedtest. Noexperimentlaunchedbythisproposal.

Reward/creditdirection remainsavailable; choosecombinednextstepfromactualinterfacepilotresults ratherthanblindrewardweightsweep. Goalunchanged.
