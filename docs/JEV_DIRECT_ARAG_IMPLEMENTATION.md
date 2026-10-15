# Full-tool A-RAG: synchronized instructions and direct dispatch

2026-10-15. This implementation restores the full tool interface; the earlier v1.1 three-action measurements are discarded and must not supply new training labels or performance conclusions.

## Implemented behavior

The new configuration is `arm=G1`, `tool_profile=multimodal_full`. It preserves sentence-level text retrieval, pixel-embedding retrieval and reciprocal-rank fusion in `semantic_search`, progressive chunk reading, search-linked image delivery, deduplication, image budgets and iterative execution. The original A-RAG exposes three retrieval interfaces; our explicit visual tools are an adaptation, not original A-RAG tools. The original repository calls the read interface `chunk_read`; this repository retains its historical adapter name `read_chunk`. [Original paper](https://arxiv.org/abs/2602.03442), [original implementation](https://github.com/Ayanami0730/arag).

The complete ordered action set is:

```text
keyword_search, semantic_search, read_chunk, image_search, figure_read, FINAL
```

`tool_registry.py` defines schemas, descriptions, action ordering and callable bindings. The effective instruction is rendered from those schemas. Schema/ordering changes invalidate the head contract. The historical `legacy` profile remains available to reproduce earlier configurations; it is not the new full-tool experiment.

`DirectHFReader` runs the complete multimodal backbone and reads its final normalized hidden state. It skips the vocabulary projection for action selection. The selected action reaches the registered runtime handler without generating a tool name, native tool prefix or call wrapper. Tool-call metadata is constructed in Python for normal A-RAG history transport; tool names still appear in the next iteration's serialized history.

Arguments still require generation. This version gives the selected operation's schema to a **separate, charged prefill** and generates only its argument JSON. It neither reuses KV across different instructions nor claims this is faster than the old shared-KV implementation. Final answer generation follows the existing answers/citations/reasoning format. Invalid or truncated tool arguments fail without executing a different tool or silently reverting to a planner. FINAL ends the direct policy; the legacy V2 verification loop does not override it.

`figure_read` now queues only requested figures, so unrelated images in a shared chunk cannot consume the budget before the selected figure. Associated chunk text remains available.

## Checkpoint and input identity

Production checkpoints require `jev-direct-head-v1`, all six actions in the same order, the exact tool contract, finite weights and matching hidden dimensions. A feature contract binds the pinned model snapshot, tokenizer/processor/template assets, preprocessing code, readout version, dtype, image/context budget and allowed system prompts. Changed prompts or old Q3/four-class heads are rejected. Benchmark-specific system additions must be included in the training/exported prompt allowlist; the base prompt alone does not authorize every benchmark.

No learned six-action checkpoint has been trained in this revision. The current linear head is a control-interface baseline, not the proposed world-model method. No measured quality or transfer gain is claimed.

Export the base contract without loading a GPU model:

```bash
PYTHONPATH=src python scripts/export_direct_tool_contract.py \
  --model-path /path/to/pinned/40-hex-snapshot \
  --output full_tool_contract.json
```

The tracked experiment runner accepts `--arm G1 --tool-profile multimodal_full --reader-mode direct-head --model-path MODEL --head NEW_CHECKPOINT`. A new, valid learned checkpoint and the corpus/index/question arguments are required. It does not train a head, reuse discarded labels or silently expand an older head.

## Validation

- 72 CPU tests passed across the repository excluding `tests/test_hf_reader.py`, whose legacy native parser requires the unavailable vLLM package. The new direct path does not use that parser.
- All-five-tool integration tests verify actual registry dispatch, next-iteration image delivery, FINAL and selected-figure budget behavior.
- The real Gemma backbone functional smoke passed: all five runtime handlers executed, both synthetic images reached the reader, FINAL JSON was valid and there were no reader/handler errors. [Raw functional report](../lab/reports/jev_v1.1/direct_arag_functional_smoke.json).
- That smoke uses a **scripted diagnostic head and synthetic fixed embedding rankings**. Actual argument generation and backbone execution are tested; learned decisions, retrieval quality and benchmark accuracy are not. Synthetic checkpoints are rejected in production mode. Shared GPU execution was authorized; no comparative latency conclusions are valid.

[Astra verification](JEV_DIRECT_ARAG_ASTRA_REVIEW.md) separates structural approval from research novelty and learned-policy performance.
