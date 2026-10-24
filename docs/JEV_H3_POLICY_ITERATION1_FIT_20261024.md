# One current-H3 action policy update from terminal branch outcomes

Hypothesis: current-policy multi-turn returns can improve four-action choices,
recovering useful exploration while retaining H3's cost efficiency. Frozen
reader and native interface; no new separate model, no PPO/GRPO or coefficient
sweep. Collection protocol remains immutable and separately hash-verified.

On completed64states/24train questions, each legal action has terminal F1 and
remaining generated cost under the SAME H3suffix. Categorical targets maximize
question-balanced weighted terminal F1 subject to weighted remaining cost<=H3's
reference cost. Each chosen state's F1 must not fall below H3's reference value;
zero-to-zero cheaper changes are excluded. Any errored pair freezes that state.
Tie-break minimum cost then fewest changes. This allows useful expensive actions
funded by positive-quality savings elsewhere, without trading away any sampled
state's quality. Overlapping suffixes are a training surrogate, not episode cost.

Require corrections in>=3training questions. Change only the existing H1MLP's
final128->4 linear layer within H3 (516parameters); freeze LLM, normalization,
MLP representation and existing H3FINAL residual. No new inference module or
architecture search. Start exactly at H3. Targets at correction states blend
.5currentH3 softmax probabilities+.5chosen-action onehot. Retention states use
currentH3 probabilities, including old features only (not their old outcomes).
Groups correction/retention get equal loss weight, each question balanced.

One full-batch100update AdamW.001/defaultweight_decay.01,clipnorm1,seed20261073.
No training-loss/dev epoch selection. Verify all non-output-layer parameters
unchanged, finite checkpoint and exact reload. One paired64development H3/new
policy evaluation after fit; no intermediate selection, heldout80 protected.
If correction coverage fails, do not fit and record missing evidence.

This is one supervised policy-improvement step from current-policy returns,
not exact value-function learning or proof of optimal multi-step planning.
Primary final judgment remains actual F1/generated and useful action changes,
fixed basicGRPO jointgoal plus separately frozen independent confirmation.
