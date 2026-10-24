# Quality-leading MLP continued with GRPO

Paired exposed validation64; same frozen reader, corpus and question IDs.

| Policy | F1 | Generated tokens | Tool calls |
|---|---:|---:|---:|
| Fixed basic GRPO | 22.50 | 98.41 | 2.33 |
| Quality MLP before | 32.45 | 139.03 | 3.41 |
| Quality MLP + GRPO | 34.75 | 150.91 | 3.70 |

The requested exact starting checkpoint was trained successfully (768 episodes,360 optimizer steps; four actor/encoder tensors changed). Paired incremental F1 +2.29pp, marginal95%CI[-3.09,+7.84]pp, generation +8.54%. Neither a reliable quality improvement nor joint efficiency success is established. Fixed-baseline target remains F1>=27.5007 and generation<=68.884375 on this development set; independent confirmation is additionally required.

The update increased exploration/cost rather than reducing cost. Do not call this early-FINAL collapse or proof of downstream policy understanding. Next bounded candidate will use refreshed current-policy outcomes and repeat-consistent quality-preserving cheaper-action preferences, retaining the reference for ambiguous states. Exact protocol must be frozen before fit; no blind coefficient sweep, no test access.

Evidence: lab/reports/decision_quality_head_grpo192_20261024/audit.json; lab/reports/decision_quality_head_grpo_validation64_20261024/report.md and campaign_comparison.json. Training and evaluation archived to Drive with hashes in their registry archive.json files. Goal remains active.
