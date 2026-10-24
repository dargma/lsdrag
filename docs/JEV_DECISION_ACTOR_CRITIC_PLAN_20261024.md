# Shared-reader multi-turn decision policy: consolidated experiment plan

Status: plan saved; no critic training, actor update, or new QA evaluation launched. User requested synthesis after approving the actor–critic direction and asking about output-token masks. This plan supersedes the pending unspecified state-discrimination hypothesis, not historical results. One candidate, at most two current-policy improvement rounds; freeze executable manifests before launch.

## Question and evidence

Can a small trainable decision module on the frozen reader improve final answer quality and generated-token cost by learning the consequences of actions over future turns?

Repeatedly exposed development64 (exploratory, not independent confirmation):

| Policy | Answer F1 (%) | Generated tokens/question | Interpretation |
|---|---:|---:|---|
| Fixed basic GRPO | 22.5007 | 98.4063 | Fixed comparator |
| H1 | 28.3042 | 75.3438 | Quality passes; cost fails |
| H3 | 27.3158 | 66.8906 | Cost passes; quality short by 0.1848 pp |
| Current-policy iteration1 | 21.2255 | 61.4375 | Lower quality; reject |

Conservative48 accepted a small parameter movement but learned 0/30 supported action corrections; no QA evaluation. H3 remains incumbent. Neither negative result proves frozen-reader impossibility. H3 constant-offset control failed its equal-budget gate on dev, so it did not establish state-conditioned superiority.

The fixed goal remains F1 >= baseline + .05 AND generated tokens <= .70 baseline, jointly, plus useful state-conditioned decisions. Current dev thresholds .2750067641 and 68.884375 are development screens only. Independent confirmation must use the basic GRPO baseline on the same frozen confirmation questions.

## One hypothesis, staged execution

| Stage | Intervention | Evidence / gate | Next action |
|---|---|---|---|
| 1. Learn future consequences | Fit one shared representation + action-quality and action-cost critics on existing H3 outcomes | Question-disjoint internal monitor: quality action-gap MSE and cost MAE both beat training-only action means; enough informative pairs | If insufficient coverage, report inconclusive; if predictive failure, do not use critic for action selection |
| 2. Improve decisions | Same encoder feeds a zero-initialized four-action residual actor over H3; conservative, cost-constrained policy update | Actual full training rollouts retain quality and budget, and supported actions change | Reject a cheap but incorrect policy; retain H3 |
| 3. Learn consequences under the updated policy | One bounded fresh current-policy branch collection, critic refresh and actor update | Actual useful continuation and correct stopping survive composition across turns | At most two policy updates total; no architecture sweep |
| 4. Judge the candidate | Freeze one candidate, paired development64 versus H3 and fixed basic GRPO | Joint goal screen plus quality-preserving savings and useful state-conditioned behavior | Only then freeze separate independent confirmation; reserved80 remains untouched until that protocol |

## Model and credit assignment

Frozen Qwen3-4B-Instruct-2507 reader; unchanged native A-RAG tool schema, argument generation, FINAL generation and shared prefill/cache. Actions keyword_search, semantic_search, read_chunk, FINAL. No extra decision LLM, reader fine-tuning, gold state features, image inputs or new retrieval corpus.

Use existing decision-prefix pack [37,2560]: pooled states plus 32 prefix bins. A small trainable encoder applies shared LayerNorm/projection to 64 dimensions, question-conditioned attention over prefix bins, then an MLP over attended context/last/question/observation summaries. Bins are compressed representations, not individual token attention. The frozen H3 actor remains an anchor. New four-action residual output initializes to zero; initialization must exactly reproduce H3 logits/actions. Shared representation feeds separate Q_F(s,a) and Q_C(s,a) output heads. This is an action-value adaptation, not literal LLaRP scalar-V PPO reproduction.

For a branch starting before decision t, execute action a and follow current policy pi to terminal:

Q_F^pi(s_t,a) target = terminal answer F1.
Q_C^pi(s_t,a) target = sum of generated argument and answer tokens from t onward.

Use observed Monte Carlo returns initially: gamma=1, no bootstrap, no invented per-turn quality reward. Quality and cost remain separate; do not fix an arbitrary accuracy/token exchange rate. Counterfactual branches describe one intervention followed by the recorded policy, not guaranteed outcomes after several policy changes.

Actor improvement maximizes predicted terminal quality subject to predicted mean suffix cost no greater than the current-policy reference, plus a reference KL constraint. Use question-balanced state weights. Freeze exact optimization, KL limit and early rejection rules in the phase-2 manifest before fitting; begin from existing KL<=.01 safety limit, not dev-tuned coefficients. At observed training states, measured branch outcomes veto actions known to reduce F1 or reward zero-quality savings. Unsupported actions require fresh outcome evidence. Critic predictions alone cannot promote a candidate. Shared encoder receives critic and actor gradients; the reader stays frozen. Every changed policy must be executed end-to-end before claiming improvement.

## Stage-1 frozen small probe

Input: /content/local_fast/lsdrag-text-v1/results/decision_h3_conservative48_inputs_20261024/states.pt; 48 training questions,128 non-forced states,460 legal-action branch outcomes. Verify parents/raw scoring/prefix hashes and input SHA before fitting. Exclude a state from supervised critic fitting if any branch has execution errors; retain exclusions in the report.

Split these 48 qids by ascending SHA256 of `20261077:` + qid: first36 critic fit, remaining12 internal monitor. All states/branches of a question stay together. This changes no official split. Existing H3/backbone training may have seen these questions; the monitor only withholds critic outcome fitting. Do not call it independent benchmark generalization.

One seed20261077; 100 full-batch AdamW updates at .001, gradient norm clip1. Quality MSE and cost MSE (cost divided by100), equal loss weights; question-balanced and legal-branch-normalized loss. Q_F sigmoid, Q_C softplus*100. Final checkpoint only; no monitor checkpoint selection or architecture sweep. Actor residual stays zero throughout this probe.

Baseline: per-action quality/cost means from the36 fit questions with the same weighting, no monitor labels. Evaluate quality-gap MSE over all legal action pairs and generated-cost MAE, question balanced. For an informative gate require at least10 unequal-F1 pairs spanning >=5 monitor questions. Pass only if both errors beat baseline and nonzero-gap sign accuracy exceeds50%; otherwise failed or insufficient evidence, not policy quality failure. Save all qids/predictions/exclusions/checkpoint/hash/metrics. The monitor gate does not claim that Q argmax is a good policy.

## Bounded later phases

After stage1 passes, one full48 critic fit and one actor update; at most one additional current-policy refresh/update. Reuse training questions only. For each refresh, preselect24 qids by hash before outcomes, fresh rollins plus all legal branches at all non-forced states, max792 episodes/1800 seconds after model load, preserving existing token/input safety caps in the frozen collector manifest. Evaluate full48 training trajectories per candidate against cached verified H3. Training gate: mean F1 >= H3, generated tokens <= H3, and at least one empirically useful action correction; report paired uncertainty, do not label train improvement generalization. No automatic dev evaluation after a failed training gate.

Only one surviving candidate gets a new development64 comparison. No coefficient tuning against dev. Required reporting: accuracy/token joint outcome; savings on quality-loss, both-zero and positive-quality-retained questions; useful costly continuation and correct stopping examples with pre-action state evidence; tool calls/input tokens and actual generated argument+answer counts. Fixed basic GRPO unchanged. A final minimal state-dependence control may be frozen after narrowing the direction; if cost is unmatched, report mechanism evidence inconclusive rather than manufacture an equal-budget claim.

## Masks and causal boundary

- Critic and actor consume decision-prefix hidden states only; future answers/results are targets, never state inputs. Existing collectors reconstruct trajectory[:turn], check message hashes and replay reference probabilities.
- Actor loss only at actual, non-forced decisions. Invalid actions masked before normalization. Forced FINAL at the cap is not a learned decision.
- Critic loss only on measured legal, error-free branches. No fabricated zero targets for illegal/unobserved actions. Padding contributes no loss.
- No token-level language-model loss in this head-only experiment. Generated argument/answer tokens still count toward Q_C and reported cost. Reader attention/causal masks remain intact.
- Terminal episodes have no bootstrap; any future TD implementation must separately distinguish terminal from truncation. Stored Monte Carlo outcomes do not need a fake next-state value.
- Standardization statistics, if introduced, use fit data only. Never put qid, gold answer, F1, suffix length or future observation in the encoder.

## LLaRP grounding and limits

Pinned official repository214c79232f726b5bf9c9c10a4bedc6044977e7df; executable isolated verification saved in lab/reports/llarp_reference_verification_20261024/verification.json, script scripts/reporting/verify_llarp_reference.py. Original scalar critic has513 parameters for512 features and receives gradients; original GAE propagates terminal reward to earlier steps. This is not full Habitat reproduction.

LLaRP trains an input bridge, actor and scalar value critic around a frozen LLM. Frozen weights do not imply fixed representations when an input bridge trains through the backbone. Our cached post-reader features differ. Its embodied task/reward/data scale do not establish that our head is sufficient for RAG, or prove simultaneous answer generation. Its paper includes subgoal reward; do not describe it as terminal-only. Pinned public code also overwrites an invalid-action penalty in SubgoalReward.update_metric; isolated execution verifies this. Do not copy reward coefficients blindly or claim code defaults exactly reproduce the paper.

Primary references: https://arxiv.org/html/2310.17722v2 ; https://llm-rl.github.io/ ; https://github.com/apple-aiml-research/ml-llarp/tree/214c79232f726b5bf9c9c10a4bedc6044977e7df .

Goal remains ACTIVE. Independent confirmation and research novelty are unproven. Save each phase transition, actual processes, raw artifacts and negative outcomes; backup Drive and push jev_v1.1.
