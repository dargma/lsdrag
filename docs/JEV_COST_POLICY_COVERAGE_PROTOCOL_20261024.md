# Current cost-policy coverage expansion

## Motivation and frozen scope

The cost-priority MLP achieved developmental F1 26.74 / 75.45 generated tokens, but its savings largely came from four still-incorrect questions ending without tools. This does not establish downstream information-value selection. The latest goal continuation explicitly authorizes bounded further experiments; the preceding status-only turn made no experimental progress. Historical plan-only holds do not authorize re-running old experiments and are not used as launch instructions here.

Collect a third disjoint batch of 48 from the remaining 96 questions in the unchanged train192. Select ascending SHA256("20261065:" + qid), before outcomes. Registry: `lab/registry/decision_cost_priority_refresh48c_20261024/manifest.json`. Frozen policy is cost-priority MLP SHA256 `e1a630e50d323b09a93f591c9394bbfe2c8fa8f5bc99f271dd1da180d55f24b7`. Initial and last nonforced state, at most two per question; each legal action receives two sampled suffixes under this same policy. Same native four-action contract, model, scorer and shared101962 corpus. Runtime receives no gold; scoring happens after response.

Limits: 816 episodes, 3600 seconds after model loading (coordinator 3900 seconds), 1M generated / 60M input tokens. Single GPU, local-fast only, no changes to reserved80 or exposed validation64. No PPO/GRPO, world model, new architecture grid or inference format changes. Three batches cover 144 distinct training questions but use different collection policies; never pool outcomes as homogeneous current-policy data.

## Evidence gates and next decision

Require complete collection, raw episode/prefix/legal/action/repeat/F1/token consistency, FP32 hidden checks and verified Drive archive before fitting. Preserve failures. Report repeat-consistent strict quality improvements, nonworse cheaper actions with positive F1, and FINAL-to-continue versus continue-to-FINAL separately, including distinct-question counts. Two suffixes are noisy evidence, not confidence bounds.

Fit protocol will be frozen after train-only inspection; do not fit on partial collection or choose questions by validation failures. Prefer retention-regularized corrections justified by completed current-policy outcomes. Keep earlier caches for feature-only retention with explicit policy provenance. Subsequent evaluation must report successful versus failed trajectory cost contributions, paired uncertainty and all four baseline/current comparisons. Development improvements alone do not meet the goal. Independent confirmation requires a separately frozen candidate/protocol; reserved80 stays protected now.

## Commands

From local-fast repository with its runtime Python and PYTHONPATH=src:
`python scripts/coordinate_decision_cost_refresh.py`

The coordinator runs collection, audits and archives only. It never automatically starts a new head fit. Runtime state: `/content/local_fast/lsdrag-text-v1/results/decision_cost_refresh_coordinator_20261024.json`.
