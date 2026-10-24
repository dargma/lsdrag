# Frozen D3-nano local action-outcome benchmark

User requested public decision-model benchmarking. One publicmodel D3nano2.21B at revision462efc366229ea268cf4eed3738b3fc4415e4817, publicweights downloaded local_fast. No incumbent or reader change.

Select32 qids by SHA256(`20261081:`+qid) from the previous72fit, then one saved H3state perqid by hash(qid:turn). Inputs from audited H3allaction96 pool, no F1-based question/state selection and no dev/test/reserved80. Preserve errorbranches rather than replacing selectedstates.

Reconstruct exact original predecision messages using originalquestion, systemprompt and observed trajectory[:turn]. Verify canonical prefix SHA256 equals stored prefix. Present those messages as D3state and ask one typed choice among only legal keyword_search/semantic_search/read_chunk/FINAL. Instructions maximize final answer correctness, avoid unnecessary calls, and continue if evidence insufficient. Never expose gold, futureturns, outcomes, H3choice or probabilities to D3. Full request includes only messages/legalaction descriptions; no arbitrary truncation. Record actualinputtokens and errors.

Lookup chosenaction's measured terminal F1 and suffixgeneratedtokens under the frozen H3continuation. Compare to H3's action on exactsame32states, plus perstate bestF1 regret and positive-quality savings breakdown. This is a one-intervention cached-suffix diagnostic, not D3closed-loop RAG or independently heldout quality. No newreaderQAepisodes. Existingreader generation/arguments in measuredbranches unchanged.

One fixedprompt, no prompttuning. First4selectedstates also reversed optionorder once (36D3requests total), mapped back by action names; descriptive flip count only. No statistical/controlled latency claim onsharedGPU. Mainruntime torch2.11+cu130/transformers5.18; card validates5.17: record actualsoftware and optionalkernel path, functionalcheck required before suite; do not call it officialbenchmarkreproduction. No new dependencies installed inmainruntime. Bound inference1800seconds, oneprocess, failonunsupported API/invalidchoices without silently guessing.

Interpretation gate: same32states meanF1>=H3 and meangenerated<=H3 with>=1positive useful correction, zero executionfailures, and permutation sample reported. Passing only motivates a separately frozen fullrollout; never promotes D3or declaresgoal. Failure informs whether generictypeddecisiontransfer helps our environment. Goal unchanged; no private API spending.
