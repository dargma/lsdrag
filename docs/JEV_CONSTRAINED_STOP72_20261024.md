# Explicitly protected stopping fit

Hypothesis: outcome-supervised129parameter stopping updates can learn useful
changes while preserving all retained training decisions, using hard constraints
instead of an allowed aggregate spillover rate. New candidate from original H3,
not a new alpha/epoch selection for rejected safe_stop72. Same72fit,26binary
quality-preserving targets and237retention states; official EM and F1 protection
and cost cap unchanged. No old failed candidate weights used.

H3 margin m_i and frozen128 features plus intercept x_i. Train delta in R129
with same binary target mixture and half correction/half retention weighting.
Minimize weighted BCE(m_i+x_i delta,target_i)+0.0005||delta||². For every retained
state require sign(m_i)*(m_i+x_i delta)>=min(|m_i|,.01), retaining original action.
Require question-weighted categorical KL(H3||new)<=.01 over all known states.
Both constraints/objective are convex in delta. One SLSQP solve from zero,
analytic derivatives,max500 iterations,ftol1e-10. No seed/threshold sweep.

Fit gate: solver success, violation<=1e-7, exact FP32 protected actions unchanged,
KL<=.0100001,>=3distinct questions with target-action adoption. Tool logits and
H1 tensors exactly unchanged, saved checkpoint reload exact. If fail: no rollout.
If pass: paired72training full rollouts using same gate and correctness-separated
cost reporting as JEV_SAFE_STOP72_20261024.md. Only a passing train result permits
one frozen development64 evaluation. State constraints apply only to recorded
training states; never claim safety/generalization at unobserved states.

Original +5pp/<=70% goal and reserved80 protection unchanged.
