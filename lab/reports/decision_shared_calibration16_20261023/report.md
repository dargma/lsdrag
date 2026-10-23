# Shared-corpus calibration16

101,962 shared public paragraphs; 16 previously exposed validation questions, frozen before execution. Three conditions/48 episodes, no generation errors. Corpus/index and source hashes in the frozen spec. No final test question was evaluated.

|Policy|F1|Mean generated tokens|Mean tool calls|Both support paragraphs retrieved|
|---|---:|---:|---:|---:|
|semantic_once|0.263158|56.8125|1.0000|0.2500|
|grpo_transfer|0.250000|101.8125|2.1250|0.1250|
|native_fast|0.312500|104.3125|2.5625|0.2500|

Original-question dense retrieval: support Recall@5=.5625; all-support coverage=.25. This establishes nontrivial retrieval difficulty, not that the corpus alone is sufficient for research claims. Native F1 is numerically higher but this small calibration sample does not establish superiority. Original head is a small-pool transfer control, not a shared-corpus GRPO baseline. Old and new absolute costs are not directly comparable; global content IDs also change argument lengths. All new conditions share IDs/tool semantics. Native-fast action decoding includes its additional measured prefix/generation costs.

Next: freeze a bounded train-only native trajectory collection and common policy initialization in this corpus, then collect new all-action outcomes and on-policy GRPO comparisons. No old-pool hidden/Q reuse. Preserve final160 for frozen final comparison. Main target remains unverified.
