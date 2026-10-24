# GRPO continuation from the F1 32.45 value-auxiliary MLP

User explicitly requested GRPO on the quality-leading head (referred to as32.35; measured validationF1=32.4509). This takes priority over another offline fit after the current refresh collection. Continue the existing refresh to preserve its data and avoid wasting GPU work; queue GRPO behind its audit/archive completion.

Exact starting checkpoint: /content/local_fast/lsdrag-text-v1/results/decision_opportunity_value_20261024/mlp_stop_value.pt, SHA0a1b3547cb22f7df5bc781a8a5b4aaa39843d3d046a4b2a28a2ef7f0df966021. This branch did NOT previously receive GRPO. Historical selected-head GRPO used the earlier d987... checkpoint, not this value-auxiliary branch.

First bounded pilot: all frozen train192×4on-policy trajectories×1round=768episodes; four clipped update epochs/minibatch32,lr1e−4,clip.2,KL.02,entropy.01,gradientclip1,seed20261059. RewardF1−.03mixedcost as previousGRPO, with zero actor advantages if allfourF1=0. Trajectory-balanced weights and fixed initial-head KL reference. Reference is never a separate inference model. Train encoder.weight/bias and policy_weight/bias; LLM, normalization and quality/cost output weights frozen. Shared representation updates can still change auxiliary predictions; do not claim those predictions remain calibrated. No critic.

Preserve FP32state storage and strict2e−4GPU probability replay. Independent audit uses matching DecisionValueHead architecture, checks frozen tensors and actor changes, counts errors/forcedFINAL and raw costs. This is fresh on-policy GRPO collection; intervened refresh branches are NOT substituted for GRPO trajectories.

Bounds:3600seconds after load/coordinator3900,1.8Mgenerated/100Minput caps,768episodes. Validate final iteration0 versus exact quality_start on exposed64×2,1200seconds runtime. No best-epoch selection or fresh-test access. Original final80/reserved80 preserved. Further rounds depend on actual quality/cost results, not automatic extrapolation.

Queue scripts/coordinate_decision_quality_head_grpo.py verifies current refresh coordinator identity while waiting, requires terminal complete, then train→audit→archive→pairedvalidation→fixed-basicGRPOcomparison→archive. Namespace decision_quality_head_grpo192_20261024. Failure stops chain for inspection; never implicit restart.

Goal remains F1+5pp andgenerated−30% jointly versus fixedbasicGRPO, plus evidence of state-appropriate decisions. Experiment completion is not goal achievement.
