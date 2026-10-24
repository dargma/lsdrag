# Broader current-H3 coverage and one conservative policy update

Update1 failed development quality despite13supported training corrections.
Seven unsupported fresh states also changed. This does not identify a unique
cause; test one next hypothesis: broader current-policy evidence with bounded
policy movement improves actual QA while preserving H3's useful behavior.

Collect remaining24of the original48H3/H1training pool, disjoint firstH3train24.
Fresh H3greedy rollin, all nonforced states and all legal action interventions,
each followed by sameH3to terminal answer. No outcome-based question selection.
Maximum24rollins+24*8*4branches=792episodes,1800seconds afterload. Original
native prompt/tools/arguments/generation limits, sharedcorpus and frozenQwen.
Exact dynamic coverage reconstructed independently from savedrollins. Errors
preserved; one sample/action, no stochastic repeats. No heldout exposure.

Combine first24and new24 only after complete raw/source/scorer/prefix/coverage/
archive verification. Keep prior failed candidate unchanged; start from H3.
Use the same current_policy_targets objective: maximize weighted terminalF1
under reference suffixcost cap, no per-state F1loss, no cheap zero-quality
changes, erroredstatesretained. Require>=3supported correction questions.

One516parameter outputlayer proposal, same loss/100steps/.001/seed20261075.
Freeze all other weights. Then backtrack along the line from H3 to proposal
using training features only: alpha=1,1/2,...,1/1024; accept the largest alpha
with mean question-balanced KL(H3||candidate)<=.01nat on all deduplicated
training features AND changed argmax on <=2% of unsupported retention states.
Coverage weights/question identities fixed before fitting. These are empirical
training limits, not guarantees on unseen states. If no nonzero update passes,
retainH3 and report no new candidate. No dev selection of alpha/thresholds.
If accepted update changes no supported correction actions, do not launch a
meaningless unchanged-policy evaluation; record missing learning progress.

Only one accepted candidate gets paireddevelopment64vsH3, max128episodes/
1800seconds. Larger coverage+conservatism are a combined direction test, not
independent causal ablation. Raw QA, useful savings, and fixedbasicjointgoal
remain decisive. No new architecture, separateLLM, backboneunfreezing or
PPO/GRPO sweep. GoalACTIVE,reserved80protected. Fit follows audited collection;
no automatic fit from partial outcomes.
