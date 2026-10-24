# Final H3 tool-return experiment: reject candidate

One frozen outcome-preference candidate and one state-independent tool-bias control were evaluated on the same repeatedly exposed development64. Native reader, argument decoding, tools and corpus were identical. This is not independent confirmation.

| Policy | F1 % | Mean generated | EM correct /64 | Correct mean tokens | Incorrect mean tokens |
|---|---:|---:|---:|---:|---:|
| Fixed basic GRPO | 22.50 | 98.41 | 10 | 92.00 | 99.59 |
| H3 | 27.32 | 66.89 | 14 | 85.86 | 61.58 |
| Constant tool-bias control | 25.75 | 66.58 | 13 | 84.54 | 62.00 |
| State-conditioned tool-return head | 23.12 | 68.66 | 12 | 80.75 | 65.87 |

The candidate changes first tool choice on26/64 episodes; all first differences occur in a tool action after identical public history, not in stop/continue at the same state. H3 is reproduced64/64 exactly in answer, F1, generated and prompt tokens. Zero runtime errors. Raw official F1/EM, per-call generated totals, frozen sources/checkpoints, immutable GRPO episode hashes and split exclusion verified. Reserved80 untouched.

Compared with H3: F1 −4.19 percentage points, mean generation +1.77 tokens;2 F1 gains and7 losses. Exact-correctness pairing:11 both correct (31 extra tokens),1 newly correct (6 saved),3 lost correct (106 extra),49 both incorrect (18 saved); total113 extra tokens. Candidate correct12 total969/mean80.75, incorrect52 total3425/mean65.8654; partial incorrect5 total699/mean139.8. Candidate is worse in both F1 and tokens than the constant control.

Original joint screen fails: versus immutable GRPO the candidate gains only0.6243pp F1 although generation is69.77% of baseline. Do not claim useful state-conditioning or goal success. Keep H3 as the prior incumbent.

Interpretation: this final small-head outcome-preference extension did not transfer on the development set. Training target fit (22 changed preferences) is insufficient evidence of closed-loop policy value. The experiment cannot establish that frozen heads, reinforcement learning, or long-horizon policies generally cannot work. It tests one fixed update from H3 continuation returns; it does not iteratively train on the new policy’s own outcomes.

Learning terminology: H3 and this candidate incorporate reinforcement-learning concepts through action interventions, terminal QA and future cost, followed by supervised fitting of hindsight preferences. Neither uses a learned critic nor PPO-style iterative on-policy training. Prefer “rollout-based policy improvement” over claiming a generic online-RL algorithm.

User requested ONE LAST experiment. This run and reporting are complete. No additional experiments or reserved confirmation are launched without renewed user instruction. Overall goal remains unachieved.

Artifacts: lab/reports/decision_tool_return_final_dev64_20261024/verification.json; frozen protocol docs/JEV_FINAL_TOOL_RETURN_20261024.md. Fit backup SHA a7f39c40a174203b081b1e07a3a35b783f4dedf5cef670c8788f20b45c8c1060; evaluation backup SHA cdca962c6e5c3a3ea43e9650e56130a7c3ca80541c2f8c094b57b1fe64f2d91e.
