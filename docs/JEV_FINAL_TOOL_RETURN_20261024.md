# Final bounded H3 tool-return experiment — 2026-10-24

User requested ONE LAST experiment after questioning asymmetric argument savings. Pointer route is deferred; no additional pointer collection/fit. After this experiment and reporting, no new candidates or automatic expansion without user instruction.

Hypothesis: when H3 elects to continue, state-dependent ranking of the three native tools learned from terminal QA and remaining generated tokens improves closed-loop QA/cost over H3 and a state-independent tool bias. This expands outcome-based action learning beyond FINAL. No changed argument generator or decoding discounts.

Use frozen existing72 train questions and existing all-legal-action branches (first action intervention, then H3 suffix). At each H3-continue state, choose highest-F1 tool among error-free alternatives with no EM decrease, no F1 decrease, positive F1, and no generated-token increase relative to reference; tie by cost then preserve reference. Otherwise retain H3. These are hindsight training preferences, not policy performance.

Freeze LLM and H3. Add one zero-initialized linear128-to3 residual (387 parameters) on H3 internal state. Recenter the three adjusted tool logits so their maximum equals H3's maximum; leave FINAL unchanged. Thus greedy stop/continue at the SAME state stays H3 exactly. Changed tool choices can create different later states and different episode lengths; no assertion of identical complete stopping trajectories.

Fit200 full-batch AdamW epochs lr.003 wd.01 seed20261088, weighted CE +.1 KL to H3 tool probabilities. Changed and retained groups each have half weight, equal question mass within group. No tune/restart/checkpoint selection. Fit a3parameter constant bias with the identical loss/epochs as the mechanism control. This is one experiment with a predefined control, not a candidate sweep.

Before evaluation verify frozen anchor equality and same-state greedy stop invariance on all184 training states. Run one paired development64 evaluation of H3, state head and constant control, using unchanged native reader/prompt/arguments/handlers. This is repeatedly exposed development, NOT independent confirmation. No development tuning or reserved80 usage. Report official F1/EM, overall generated tokens, per-policy correct/incorrect/partial costs, paired correctness costs, and state versus constant comparison. Compare unchanged basicGRPO F1 .22500676406926406 and mean generated98.40625; joint screen F1>=.27500676406926406 and mean generated<=68.884375. Passing this screen alone does not prove goal completion; independent confirmation would need a separate explicit frozen plan.

This fits one policy-improvement step from H3 suffix returns, then evaluates recursive execution of the new policy. It does not establish optimal multistep planning, on-policy convergence, or calibrated value estimation. No critic is introduced.
