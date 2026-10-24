Status override: DEFERRED after user fairness objection. Collection completed with insufficient fit coverage (8 < 12 read questions). No pointer fit or additional collection authorized by this plan. See final H3 tool-return protocol.

# H2 native read-pointer feasibility — 2026-10-24

Status: native H2 reference collection frozen and launched (2026-10-24); no pointer fit yet. Collection specification: lab/registry/decision_h2_read48_20261024/spec.json, SHA256 fda7a5709e66d13b79334653670477e6d313a87900e8f0d484c7cc09a478c837.

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

## Frozen collection implementation

ReadCaptureReader passively hooks the existing native prefill once, retaining its final-layer hidden states until the chosen action is known. For reads, it stores mean contextual features of complete latest observed document objects, ordered by most recent result and deduplicated by ID, plus the existing state pack. Exact tokenizer offset roundtrip is required; missing observations fail explicitly. Native argument decoding, payloads and handler remain unchanged. Model/reader/checkpoint/source hashes are frozen in the collection specification. Runtime PID82714/session43644 was confirmed live at launch; use actual status/process checks, not these historical identifiers.
