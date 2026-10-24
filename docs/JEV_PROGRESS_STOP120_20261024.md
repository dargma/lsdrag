# Exploration-progress stopping hypothesis

Constrained72/120 fit useful training corrections but lose quality on exposed
development. Next test one representation hypothesis, not another data/solver
sweep: can causal exploration progress support transferable stopping decisions?

Use original H3 as a frozen anchor and its128-dimensional hidden state z_t.
Add132 zero-initialized FINAL-only weights on [z_t-z_(t-1), one-hot previous tool
(3), t/8]. At t=0 set the entire feature vector to zero: the first decision is
exactly H3 even after training. There is no new learned intercept. Previous state
and action come only from this episode, reset through the existing evaluator hook.
Cache is transient, not checkpoint state. All tool logits and reader generation
path remain unchanged. No new LLM calls. This is a minimal progress representation;
new evidence IDs, repetition metrics and spent-token features are deferred.

Inputs: same fixed120 fit questions/currentH3 outcomes, all322 recorded states.
Each noninitial state must have its immediate predecessor in the same H3 rollin;
initial feature zero. No fabricated predecessor or future observation. Separate
24monitors/development/reserved remain excluded. Binary stop/bestcontinue targets
use terminal F1 and officialEM nondecrease and weighted suffix-cost cap as before,
with initial states fixed to H3 in the target optimization. Learning label is
future outcome, not novelty itself. Prior retention features lacking verified
causal predecessors are not reused; protect every uncorrected currentH3 state.

One fixed convex fit with existing fit_constrained_stop: half correction/half
retention BCE, same mixture targets, L2 .0005||delta||², hard retained-action
constraints, question-weighted categoricalKL<=.01, SLSQP500/ftol1e-10. Same
fit gate: solver success, exact protected actions, KL<=.0100001,>=3changed
questions with supported targets; first decision equals H3 and tool logits fixed.
Then fixed fulltrain120 gate and one dev64 only if it passes, with official
correct/incorrect costs and no lostcorrect training question. No seed grid.

Runtime profile: full greedy episodes with no first-action intervention/native
selection override. The module stores its greedy action as the next step's prior
action; do not silently use it with stochastic or externally forced actions.
Test zero initial residual, causal feature contents, tool-logit invariance and
cross-question reset before fitting. Source/data/checkpoint hashes must be frozen
in a new registry plan before the fit. Implementation tests are not policy results.
Original +5pp/<=70% goal unchanged. Reserved80 requires separate confirmation.
