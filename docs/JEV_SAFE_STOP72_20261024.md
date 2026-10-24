# One quality-preserving H3 stopping update

Hypothesis: H3's current stopping boundary can improve without trading away
per-state measured answer quality. Same72fit questions as prior shared actor;
24monitor and development/test/reserved excluded. Use existing all-action H3
suffix outcomes, restrict candidates to FINAL and H3's best non-FINAL action.
Official EM labels rescored from saved raw answers. Permit a switch only with
F1 and EM nondecrease and positive F1. Question-balanced MILP maximizes F1 under
current weighted suffix generated cost, then minimizes cost and changes.
No zero/zero cheap targets. Errors retain reference. These targets are hindsight
training surrogates, not a guarantee when policy changes compose.

Initialize from H3, freeze H1 backbone/head, update129 FINAL parameters only.
Same prior72-only old features plus uncorrected new states for retention.
Margin BCE .5reference+.5target; half correction/half retention loss,
question-balanced each.100AdamW updates lr.001 decay.01 clip1 seed20261083.
Backtrack parameter delta1,1/2,...1/1024 using training-only categoricalKL<=.01,
unsupported action changes<=.02. First acceptable step fixed; require>=3distinct
questions with learned target actions. Failed gate means no QA rollout, no
hyperparameter retune. Tool logits stay exactly fixed at the same state.

If learning gate passes, paired72 full training rollout H3/candidate. Require
F1>=H3, EM>=H3, generated<=H3, at least one strict aggregate improvement and
positive savings among jointly correct questions or newly correct questions.
No reference-correct/candidate-wrong questions allowed at this train gate.
Report counts/totals/means for exact-correct/incorrect and partialF1, plus paired
four-way correctness groups. Only a passing result enables one frozen dev64
comparison. Dev is repeatedly exposed, never independent confirmation. Original
+5pp/<=70% goal and reserved80 protection remain unchanged.
