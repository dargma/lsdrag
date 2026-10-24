# H2 native read-pointer feasibility — 2026-10-24

Status: proposed bounded pilot; no model training or collection launched.

Objective remains F1 +5 percentage points and generated tokens <=70% of immutable basic GRPO jointly, useful state-dependent policy evidence, independent frozen confirmation. Reserved80 remains untouched.

## Evidence and hypothesis

Train-only audit of 120 existing H3 reference episodes found 23 read calls in 21 questions. Exact legacy four-slot membership covers 11 calls / 358 of 1112 read-generated tokens (32.2%). Removing the four-slot cap while retaining result groups and singletons covers 14 / 446 tokens (40.1%). An ordered pointer over observed IDs can represent all 23 payloads. These are representational upper bounds, not prediction quality, H2 coverage, or actual savings. The sparse H3 sample is insufficient to claim H2 feasibility.

Hypothesis: retaining quality-leading H2 action policy while replacing long opaque-ID autoregressive decoding with an ordered observed-document pointer can preserve QA and avoid enough real decoding to pass the original joint screen.

## One bounded next experiment

1. Freeze 48 training questions by hash seed 20261087 from the existing 120 train pool, independent of outcomes. No development labels or reserved80. Generate H2 reference episodes and capture pre-read hidden features plus observed candidate IDs and their causal text positions. H2 checkpoint, source, prompt and split hashes must be frozen before launch. Native argument generation supplies teacher labels; no gold enters runtime.
2. Split these questions deterministically 32 fit / 16 diagnostic before collection. Measure read coverage first. If fewer than 12 fit questions and 6 diagnostic questions contain reads, do not fit a purported general policy; report inadequate coverage and review collection scope.
3. Train one small ordered pointer sharing the frozen reader's contextual features. Candidate representation must be tied to the observed document, not opaque-ID spelling or fixed position alone. Predict document indexes plus STOP; map back to the exact native chunk_ids list, preserving order. No additional reader invocation. Keep existing action policy unchanged. Use generation fallback when the train-calibrated confidence gate declines selection. Freeze architecture, feature extraction and confidence rule before fitting; no diagnostic tuning.
4. On diagnostic questions compare H2 always-generate against H2 pointer/fallback using actual full rollouts. Check selected payload/order and actual decoding avoided; report prompt tokens, generated tokens, pointer compute, errors, final official F1/EM and correct/incorrect costs separately. Do not count merely copied or renamed previously decoded tokens as savings. No controlled latency claim on shared GPU.
5. Continue only if diagnostic QA is preserved with useful real token reduction. Original goal applies unchanged at subsequent frozen screening and independent confirmation; passing this pilot alone never proves the goal or a future-aware document policy.

## Research claim boundary

Teacher-ID imitation establishes efficient argument execution only. If feasible, a separate bounded outcome-training step must compare alternative document choices followed by the current policy to terminal answer, using final QA and remaining cost. That later step can test future-return-dependent document choice. It is not authorized by this document to silently expand collection or consume reserved80.

Next executable preparation: inspect H2 loader and shared-reader capture format, implement causal candidate feature extraction and freeze the 48-question collection specification. Existing old per-question-corpus slot checkpoints are historical and cannot substitute for this shared-corpus experiment.
