# Full six-action head pilot

Status: **teacher_in_progress**. Snapshot: 2026-10-15T10:09:53.414994+00:00

This is fresh-label behavior cloning of a generated teacher, with no world model. It does not demonstrate better decisions or answer quality. Discarded three-action results are not used. The GPU is shared, so this report makes no latency comparison.

Completed teacher questions: **12**; natural labelled states: **1**. EM: 0.0; F1: 0.0; valid answers: 0/12.

Episode statuses: `{"failed": 12}`. Errors: 12 across 12 questions; selector/format errors: 12. Delivered images: 0 episode-summed unique images across 0 questions.

## Natural action coverage

| Action | States | Distinct questions |
|---|---:|---:|
| keyword_search | 1 | 1 |
| semantic_search | 0 | 0 |
| read_chunk | 0 | 0 |
| image_search | 0 | 0 |
| figure_read | 0 | 0 |
| FINAL | 0 | 0 |

Forced FINAL records excluded: 0. Natural FINAL is retained. A20-question block is diagnostic. Training requires at least60 completed questions and the separate natural six-action coverage gate; a missing phase is not a passed gate.

## Training and held-out imitation

| Seed | Training available | Final train loss | Held-out agreement | Dev completed |
|---|---|---:|---|---:|
| 20261015 | False | None | not available | 0 |
| 20261016 | False | None | not available | 0 |
| 20261017 | False | None | not available | 0 |

All preregistered seeds are shown; no best seed is selected. Held-out teacher agreement measures imitation on the same generated-teacher states and is not task success.

## Scope and integrity

Public M3DocVQA/MMQA question lineage with fixed previously used288-document shared corpus.

51 of full100 training questions overlap prior100; development0 prior100 qid overlap. Not strict unseen-source or final-test evaluation.

Run manifest SHA256: `9537877e324665ff3170568b239975b1a60cf700e6bb6f7e26fa69f5adb3eac3`. Data manifest SHA256: `f485ebf6901a669938dc761ce1050b0a82f8050a21cd242cd8cecd011fab47f7`. The JSON includes source/data hashes, artifact hashes, gates, phase summaries, incomplete records and provenance paths.

- Fewer than60 completed teacher questions; initial20 is diagnostic, not a head-training success.
- One or more natural actions have fewer than5 distinct teacher questions; no six-action coverage success may be claimed.
- Held-out evaluation is incomplete or not yet run; no held-out success claim.
