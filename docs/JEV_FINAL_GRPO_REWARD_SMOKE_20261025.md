# Last GRPO proposal: correctness + generated length + format

User: one genuinely last experiment; latest request first run one smoke and save carefully. This execution is restricted to the smoke. No automatic main training, evaluation grid or coefficient sweep.

Starting policy:9B value_aux checkpoint5191c60eee742695031856b2c66a2b3581995b65ffa78b6347cc9a84d9b476d7. FrozenQwen3.5-9B reader/sharedencoder; train the existing small actionMLP net.* only. No separate critic and no LLM token-gradient training. Arguments/answers remain greedy under unchanged grammar. Four legal action choices are sampled from the actual current head without epsilon mixing.

Reward per completed trajectory:

R = EM + 0.25 F1 - 0.10 T/(T+256) + 0.02 I(format valid).

EM/F1 are official normalized answer/alias metrics. T is actual emitted argument+final tokens, including wrapper-boundary tokens, excluding forced input prefixes; input counts are also recorded. The bounded negative length term strictly prefers shorter trajectories for equal accuracy/format; exact correctness dominates its full range. No positive short-answer bonus or replacement of total trajectory length by final answer length. All-wrong groups can still favor shorter wrong trajectories: this is a known risk, not masked away. Format criterion: every actual raw argument/FINAL JSON passes unchanged per-action schema, generation finished schema_complete, episode terminates withFINAL and no execution errors. Existing generator constrains grammar, so reward may be constant. Constant format bonuses cancel in within-question GRPO normalization; report its variation rather than claiming it trained formatting.

Final schema: `{"answer":"short answer span","support_ids":["observed supporting chunk ID"]}`. Native tool wrapper is assembled by runtime; do not reward an invented XML/think format or alter the generator. Format validity does not establish answer correctness or supporting evidence truth.

Smoke: first2 frozen outcome_train48 IDs, chosen by existing order (not correctness),4sampled paths each;2greedy before+2greedy after on SAME training questions. Total12episodes, no dev/reserved/final access. FP32 saved packs, behavior logprob replay atol/rtol2e-4. Group advantages=(R-mean)/(std+1e-6), trajectory-balanced action loss, clippedratio.2, KL.02 to startinghead, no entropy bonus;2epochs minibatch16 AdamW1e-5 clipgrad1 seed20261085. One update phase only, finalcheckpoint. max1200seconds/100000generated;>=40GiBfree beforeloading. ForcedFINAL contributes tokens/reward but is excluded from policy gradient.

Pass: actual12episodes complete, official reward andschema recompute, token sums, finite checkpoint, frozen normalization/reader, actualheadweightchanges, behavior replay pass, groupvariationreported, raw/batch/checkpoint backups verified. No quality claim from2trainingquestions; smoke pass does not prove main experiment succeeds. If allgroupadvantages zero, record no reward signal rather than claim RL learning from weight decay. Do not repeat smoke with cherry-picked questions.

Runner `scripts/run_qwen35_grpo_reward_smoke.py`; frozenplan `lab/registry/qwen35_grpo_reward_smoke_20261025/plan.json`; runtime `/content/local_fast/lsdrag-text-v1/results/qwen35_grpo_reward_smoke_20261025`. Historical checkpoints/sources preserved. Main experiment remains unlaunched; smoke checkpoint is not promoted or silently used to start a main run.
