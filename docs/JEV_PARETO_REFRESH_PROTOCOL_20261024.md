# Refreshed empirical quality-preserving action preferences

One bounded offline candidate, frozen before fitting. Start exact quality MLP0a1b (F132.45); its suffix policy generated the refreshed48subset of frozen train192. Preserve latest GRPOc6b0 (F134.75) separately. No backbone update, no validation labels or reserved test access.

Preference a over b only if each has2error-free samples, minF1(a)>0, minF1(a)>=maxF1(b), and maxGenerated(a)<minGenerated(b). This empirical separation is not statistical confidence or guaranteed dominance. Observed32states/105pairs. Exclude allzeroquality preference. Do not mix oldBCoutcome targets; old377states only retain anchor probabilities across train192.

Train encoder and policy output only, frozen auxiliary outputweights/normalization.100epochs AdamW1e-4,gradientclip1,seed20261060. Pairwise softplus of negative logitdifference, question/state balanced; plus0.1 times average reference-to-current KL onfresh88/old377states. No reward coefficient sweep, no direct zero-cost FINAL heuristic. Shared encoder changes can alter auxiliary predictions despite frozen outputweights.

Three selection-rule tests passed. Allsource/inputhashes in registryplan. Check every tensor finite and nonactor tensors unchanged. One final checkpoint, no epoch selection. Evaluation same64×2quality_start/pareto, original corpus/model/prompt/runtime unchanged, max1200seconds and128episodes; compare fixedbasiccachedreference too. Goal remains active regardless dev result; independent confirmation deferred. If actual quality collapses despite empirical preferences, reject transfer rather than claim constraints guarantee safety.
