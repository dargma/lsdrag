# Shared-corpus policy-head methods and reporting contract

This describes the implemented shared-corpus restart, not a claim of successful improvement. The frozen experiment manifests and raw artifacts establish what actually ran. The original eight-hour window ended before this restart's continued execution; the user explicitly resumed work and each new stage has a bounded budget.

## Model and state

The non-thinking Qwen3-4B-Instruct-2507 backbone is frozen. A small head maps its prefill representation to four legal-action probabilities: keyword_search, semantic_search, read_chunk, FINAL. The same backbone generates tool arguments and the final answer. There is no independent decision LLM or learned judge. The head itself does not generate language. Arguments are generated in these experiments; previous argument-reuse profiles are not part of this comparison.

Every question searches the same 101,962 deduplicated public paragraphs. Search returns titles and the first300 characters, default top-k5 and maximum20. Original keyword occurrence scoring and dense dot-product scoring are preserved. Corpus IDs are content-derived; model-visible fields are ID/title/text only. Source questions, answers, support flags and decompositions are not corpus fields. Public evaluation documents are searchable; evaluation question labels are not training examples. This is dataset-derived shared-corpus retrieval, not full Wikipedia.

The original head uses last-token hidden, train-only mean/scale normalization, an MLP2560→128(tanh)→4 plus a linear skip2560→4. Shared BC initialization learns native action IDs from train192,667 non-forced states;100epochs,AdamW3e-4,question-balanced CE,.05 smoothing over legal actions only. RL and original all-action improvement update action and skip layers, holding the representation trunk fixed. Critic/success/cost outputs are unused. A separate multi-position MLP candidate concatenates last/question-end/last-observation-mean. It has its own matched BC/outcome pair; capacity, trainability and KL anchor also change, so it is not a pure input-location ablation.

## Costs and trajectory reward

For each decision step, measured decoded tokens g_t and all accounted input/prefix tokens p_t define the proxy

\[
c_t=\mathbf1[a_t\ne\mathrm{FINAL}]+g_t/100+p_t/10000,
\qquad G(\tau)=F1(y,\hat y)-\lambda\sum_t c_t.
\]

GRPO uses lambda=.03. All-action candidates use .03 or .06. Cost is a proxy, not latency. Generated-token totals include tool arguments and final answers; forced syntax prefix tokens are separately recorded. Native-fast action generation incurs its actual additional prefix/generation costs. Full trajectory costs include any cap-forced final step; cap-forced action choices are excluded from policy learning. Errors remain in the data and cost totals.

## Online GRPO

Collect K=4 trajectories for each of Q=192 training questions under the current stochastic head (epsilon0), then update. Two rounds are planned. Each round uses rewards

\[
A_{qk}=\frac{G_{qk}-\overline G_q}{\operatorname{std}(G_{q1},\dots,G_{qK})+10^{-6}}.
\]

The same trajectory advantage applies to each of its controllable action decisions. With N stored decisions and T_qk controllable decisions in that trajectory,

\[
w_{qkt}=N/(QKT_{qk}),\qquad
r_t(\theta)=\exp(\log\pi_\theta(a_t\mid s_t)-\log\pi_{old}(a_t\mid s_t)),
\]

\[
L=\mathbb E_{batch}\left[w_t\left(-\min(r_tA_t,\operatorname{clip}(r_t,.8,1.2)A_t)
+.02D_{KL}(\pi_\theta(\cdot\mid s_t)\|\pi_0(\cdot\mid s_t))-.01H(\pi_\theta(\cdot\mid s_t))\right)\right].
\]

Here pi_0 is the frozen common BC initialization, not a separately trained judge. Implementation:4epochs per round,minibatch32,AdamW1e-4,gradient norm clip1. Worker verifies behavior log-probability replay before each update. Group standardization can amplify small cost differences when correctness is tied; report actual correctness/cost tradeoffs instead of assuming reward improvement means accuracy improvement.

## Same-state all-action improvement

Under the frozen common BC policy, take the first and last controllable public states of each training trajectory. For every legal action, force that first action and sample two suffixes under the same frozen policy. Match prefix hashes, hidden input and action legality across interventions. The forced branches are never treated as on-policy GRPO samples.

\[
\widehat Q_\lambda(s,a)=\tfrac12\sum_{j=1}^2\left(F1_{s,a,j}-\lambda C_{s,a,j}\right),
\qquad V_{ref}(s)=\sum_a\pi_{ref}(a\mid s)\widehat Q_\lambda(s,a).
\]

For question-balanced state weights alpha_s, optimize

\[
L=\sum_s\alpha_s\left[-\sum_{a\in\mathcal A_s}\pi_\theta(a\mid s)(\widehat Q_\lambda(s,a)-V_{ref}(s))
+.02D_{KL}(\pi_\theta\|\pi_{ref})-.01H(\pi_\theta)\right].
\]

All-action expectation uses the full probability vector, not only the winning label. Q targets are detached Monte Carlo outcomes, without group standard-deviation normalization.100full-batch epochs,AdamW1e-4,gradient clip1. A state-only baseline leaves the exact all-action gradient unchanged. Two suffixes per action give noisy estimates; this is a policy-improvement surrogate, not a calibrated value model or guaranteed policy improvement. The structural candidate uses these same Q targets but its own BC KL anchor, introducing an additional off-policy approximation.

## Evaluation and claim limits

Training192,validation64 and final160 question/decomposition-family splits remain frozen. Validation64 is reported as original32 and additional32 as well as overall. At most two learned candidates are selected before final outcomes, using the predeclared joint-target score. Final baselines include shared-corpus GRPO, common BC, native-fast, semantic-once and the matched multi-MLP warmstart. F1+5percentage points and generated-token reduction30% are judged against the shared-corpus GRPO on the same final questions. Also report whether the candidate is useful relative to native and the simple workflow; beating a weak GRPO reference alone is insufficient evidence of a strong contribution.

Paired-question bootstrap intervals are conditional on frozen checkpoints and do not estimate training-seed uncertainty. Current shared-corpus training uses one seed per declared run; earlier small-pool seed replication does not establish stability here. Original and structural candidate initialization differences are disclosed. GRPO1536 training episodes and all-action up to3264 episodes are not compute matched; include all collection, calibration, validation and candidate-search costs. Do not equate token reduction with wall-time speedup on a shared GPU. Frozen policies, corpus/source hashes and original failed outcomes remain auditable. No test-based parameter tuning or world-model expansion is part of this execution.
