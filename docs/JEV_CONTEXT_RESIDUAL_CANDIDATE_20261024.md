# Optional context-bin residual input extension

Candidate implementation only: src/lsdrag/decision_context_residual.py. No training, validation or GPU launch. Current refresh and explicitly requested quality-head GRPO keep priority and their frozen sources remain unchanged.

Verified current pack_hidden contract: slots0..4 are last token, question end, observation end, last16mean and observation mean; slots5..36 are32ordered disjoint full-prefix bin means. They are NOT individual observation tokens. Existing historical PolicyHead already includes a32-bin attention option; this is not a novelty claim or newly discovered feature type.

New controlled candidate preserves the actual quality-leading MLP exactly at initialization: copy and freeze its parameters, then add a zero-output-initialized action-conditioned context residual. Keys/values read the32cached bins; queries combine question-end representation with learned action queries. All attention parameters are within the head; no backbone change. Train-only bin normalization. Base quality/cost outputs stay unchanged because the base is frozen, but policy changes can invalidate their interpretation as future outcomes.

This avoids the earlier attention candidate's distillation mismatch at initialization while providing access beyond3summaries. It still changes head capacity and computation and uses pooled spans; no claim of token-level selection or equal cost. Normalization/statistics must be fit on the final allowed training set before any actual experiment.

Two new tests verify exactinitial logits, legal masks, attention normalization, gradient flow/context sensitivity, frozenbase and invalidpack rejection; combined with quality-GRPO initialization regression3tests passed. These are synthetic implementation checks, not behavioral improvement.

If GRPO plus refreshed outcomes still leave an input bottleneck, freeze a small matched residual-head versus current-head training comparison using existing37slot caches. Do not silently insert this module into the running policy or claim it is already being evaluated. True individual-token access would require a separate reader/cache protocol.
