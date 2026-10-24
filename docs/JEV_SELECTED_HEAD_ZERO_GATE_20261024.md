# Selected-head GRPO: zero-quality group gate

User approved the proposed next experiment with “ok”. One change relative to completed selected-head GRPO r1: if all four trajectories for a training question have F1=0, set their policy advantages to zero. Otherwise keep original group-standardized F1−0.03 cost advantages. Keep KL .02 and entropy .01 active for every group, including gated groups. This does not force search or prevent FINAL at inference. It cannot teach recovery on groups with no successful trajectory.

A_i = 0 if max_j F1_j = 0; otherwise A_i = (R_i−mean(R))/(std(R)+1e−6), R_i=F1_i−0.03 C_i.

Start: exact pre-GRPO multi_mlp_outcome checkpoint d987142ff7ea00efe06748fa205e84095a500abc675f290f449e035e70eacb99. Frozen Qwen3-4B-Instruct-2507, train-only normalization fixed, all selected-head MLP parameters trainable. Same seed20261055, train192×4×2, LR1e−4, four epochs/round, minibatch32, clip.2, gradient clip1. FP32 state storage and strict GPU replay2e−4 retained. No new structure, auxiliary predictor, reward weight sweep or world model.

Shared101962-document corpus and train192/validation64 stay frozen. Original final80 and reserved80 remain unused. Development validation only: final iteration1 versus pre-GRPO head, then compare to cached ungated r1 on identical64 questions. No checkpoint selection. Report F1, generated and input tokens, tool calls, zero-tool episodes, errors, paired uncertainty. Primary intervention comparison is gated versus ungated selected-head GRPO; research target remains +5pp F1 and generated ratio≤.70 versus basic shared GRPO. These are separate judgments.

Budget:1536 training episodes,3600seconds after load,1.8M generated/100M input token caps; validation128episodes/1200seconds. Single GPU serialized lock. Failed runs retained. Coordinator performs training, independent probability/advantage/normalizer audit, verified Drive backup, paired evaluation and report/backup. No further automatic sweep.

Prior diagnostic motivation (observational, not causal proof): validation zero-tool episodes18→33; five F1-declining questions all became zero-tool. Training round0 had82/192 all-zero-F1 groups,68 with cost variation; round1 had75 and56. Mixed reward can prefer cheap failure in these groups. Gate hypothesis: remove that gradient while retaining useful cost preference where some answer quality is observed. It may increase tokens without improving F1.

Commands: PYTHONPATH=src OPENBLAS_NUM_THREADS=2 /content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/coordinate_decision_selected_head_grpo_zero_gate.py

Runtime: /content/local_fast/lsdrag-text-v1/results/decision_selected_head_grpo192_20261024_zero_gate
Registry: lab/registry/decision_selected_head_grpo192_20261024_zero_gate
Tests:11 passed (gate edge cases, selected initialization/storage replay, GRPO and rule regressions).
