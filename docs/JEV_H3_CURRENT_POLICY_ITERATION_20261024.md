# H3 current-policy multi-turn action improvement: first bounded iteration

User affirms that learning actions from current-policy multi-turn consequences
is the core research direction. Keep shared frozen Qwen/native interface.
H3 checkpoint936e4592 supplies both rollin and every branch's subsequent policy.

Reuse24fixed training H3 episodes already generated during token calibration;
these questions are not selected by F1 or policy wins. Freeze all64nonforced
states and231legal actions across them. At each state reconstruct only public
history, force one legal keyword/semantic/read/FINAL action, then run H3 greedy
to the final answer. One execution per action, no stochastic repetitions,
231new episodes maximum/1800seconds after loading. Original parent episode
hashes, source/model/corpus/checkpoint hashes and exact state indices frozen.

Unlike H3's two-action H1suffix labels, this covers all four legal actions under
H3's own continuation policy. Measure terminal F1 and all remaining generated
cost. Neither one-step retrieval score nor action imitation is the objective.
Intervention differences can reflect query/argument generation as well as tool
identity because the same frozen reader generates arguments conditionally.

Require original prefix hashes, legal masks, H3 probability replay, exact
64states/231episodes, full scorer/count checks, training split exclusion and
verified Drive archive. Preserve errors as failures, not wins. Source snapshots
must enumerate every selected turn; do not reuse old first/middle/last helper.

After complete verification freeze one action-head update using train-only
branch outcomes, retaining original behavior where evidence is unsupported.
Reader and generation semantics stay frozen; separate decision LLM, world model,
PPO/GRPO sweep, new interface and heldout access excluded. Exact target/objective
and update settings are not yet frozen: choose them from completed training
coverage, not dev outcomes. Evaluate one candidate against H3 on development64
before any further iteration. Costs and errors of collection reported separately.

This is policy improvement using returns under a fixed current policy, not a
claim of optimal planning or exact value learning. It is the first step toward
iterative long-horizon learning; one iteration does not establish convergence.
Goal remains fixed basicGRPO F1+5pp/generated≤70% plus useful state-conditioned
selection and separately frozen independent confirmation. Reserved80 untouched.
