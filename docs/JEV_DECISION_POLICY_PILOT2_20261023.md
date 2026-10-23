# Decision policy pilot2 — 2026-10-23

User authorized followup with “그래 시작해”. Main backbone remains frozen Qwen3-4B-Instruct-2507 (non-thinking). Candidate Qwen3.8 download and thinking-OFF smoke are separate; no backbone substitution.

Train-only audit found saturated BC probabilities and weak alternative-action gradient responsibility in epsilon-mixture PPO. This is a mechanism diagnosis, not proof of a sole cause. First temperature in [1,2,4,8,16] with train median maximum probability <= .8 is 8; it preserves greedy initial choices.

Frozen manifest: lab/registry/decision_policy_pilot2_20261023/manifest.json. Existing train24/validation8; new family-disjoint dev40 frozen before outcomes; exposed pilot1 test20 and earlier100 excluded from tuning. Two rounds collect one exploratory trajectory per train question and first/middle/last controllable states, including voluntary stopping. Each legal action receives two suffix rollouts under a common frozen reference. Round0 reference is tempered BC; round1 is round0 MLP quality policy.

Six learners share exactly the same per-round samples: MLP return-based improvement and outcome CE, each with quality-only or quality-minus-.03-cost objectives; linear return-based and CE cost controls. Cost is tool calls + generated tokens/100 + input tokens/10000. Equal question weighting; 100 epochs, AdamW .001, anchor KL .02, entropy .01. This is all-action Monte Carlo approximate policy improvement, NOT PPO or an unbiased on-policy gradient. Collection is shared and reference-driven by the quality PG learner, not independently on-policy for each learner.

Evaluate native, historical BC and all six final learners on validation8 and test40 without validation-selected checkpoint or coefficient. Primary comparison: MLP cost policy vs MLP cost CE and historical BC. Report QA, calls, generated tokens, cap-forced stopping, errors and paired uncertainty. Single seed; no controlled latency or calibrated success probability claim. Head output probabilities mean action-selection probabilities.

Bounds: 2 rounds, <=1152 branch episodes, 48 collectors, 384 evaluation episodes; 60 minutes excluding model load, 1M generated and 16M input tokens, 8 tool calls, 512 output tokens, context16384. CPU verification: 11 tests passed. Existing pilot1 source/results remain unchanged.

Run: `PYTHONPATH=src /content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/run_decision_policy_followup.py --output /content/local_fast/lsdrag-text-v1/results/decision_policy_20261023_pilot2`
Log: `/content/local_fast/lsdrag-text-v1/logs/decision_policy_20261023_pilot2.log`
