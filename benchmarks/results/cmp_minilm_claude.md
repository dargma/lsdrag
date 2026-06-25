# BENCHMARK — 독립 골드셋 21문항 (Reader provider = claude_code)

> 문서: ARMv8-Reference-Manual (expanded parse, data/ir_full). 검색 Recall(출처 검색율) + Reader 최종 답변 F1/EM(SQuAD식).
> 이 셋은 모델-비의존(gold=원문) → **Reader LLM을 바꿔 같은 셋으로 비교**할 수 있다.


## 집계

| 지표 | 값 |
|------|----|
| 검색 Recall (gold 출처 검색율) | **95.24%** |
| 답변 F1 (토큰, gold max) | **0.441** |
| 답변 EM (완전/포함 일치) | **57.14%** |
| 평균 LLM 호출 수 (루프) | **2.52** |
| 평균 답변 시간 (초) | **13.74** |


## 카테고리별 (컨텍스트: 표/그림/텍스트)

| category | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| concept | 3 | 100% | 0.43 | 67% |
| figure | 3 | 100% | 0.21 | 33% |
| figure-loc | 3 | 100% | 0.50 | 33% |
| heading-loc | 3 | 100% | 1.00 | 100% |
| negative | 3 | 100% | 0.31 | 67% |
| table | 3 | 100% | 0.30 | 67% |
| table-loc | 3 | 67% | 0.33 | 33% |

## 도구별 (의도 도구 커버리지)

| tool | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| any | 3 | 100% | 0.31 | 67% |
| image_read | 3 | 100% | 0.21 | 33% |
| keyword_search | 5 | 80% | 0.35 | 40% |
| page_index_search | 6 | 100% | 0.75 | 67% |
| read_chunk | 1 | 100% | 0.17 | 100% |
| semantic_search | 3 | 100% | 0.43 | 67% |

## 문항별

| ID | cat | tool | Recall | F1 | EM | 답변(발췌) |
|----|----|----|:-:|:-:|:-:|----|
| E001 | figure-loc | page_index_search | 1 | 1.00 | 1 | D1-1899 |
| E002 | figure-loc | page_index_search | 1 | 0.00 | 0 | {"final": "arm_p1981_2070:1588 is the chunk, but I need the printed page label — |
| E003 | figure-loc | page_index_search | 1 | 0.50 | 0 | D7-2198 |
| E016 | table-loc | keyword_search | 1 | 1.00 | 1 | D5-2150 |
| E017 | table-loc | keyword_search | 1 | 0.00 | 0 | Page E2-2804 — the table containing 'Support for FCSE.' Wait, no printed page la |
| E018 | table-loc | keyword_search | 0 | 0.00 | 0 | p2521 |
| E024 | heading-loc | page_index_search | 1 | 1.00 | 1 | D1-1891 |
| E025 | heading-loc | page_index_search | 1 | 1.00 | 1 | D3-1991 |
| E026 | heading-loc | page_index_search | 1 | 1.00 | 1 | D4-2090 |
| E040 | concept | semantic_search | 1 | 1.00 | 1 | D1.20 Interprocessing |
| E041 | concept | semantic_search | 1 | 0.00 | 0 | Watchpoint mismatch (no match) — page E2-2804. |
| E042 | concept | semantic_search | 1 | 0.29 | 1 | Synchronous external abort on an instruction fetch is taken precisely using the  |
| E065 | table | keyword_search | 1 | 0.00 | 0 | arm_p1891_1980:4 |
| E066 | table | read_chunk | 1 | 0.17 | 1 | Outer Shareable (per Table on page arm_p2071_2160:599 — stage 1 Non-shareable +  |
| E067 | table | keyword_search | 1 | 0.74 | 1 | 1 indicates the Overflow floating-point exception occurred during execution of t |
| E082 | figure | image_read | 1 | 0.18 | 0 | EL3 — the Secure monitor operates at Exception Level 3 (EL3). [Note: based on st |
| E083 | figure | image_read | 1 | 0.38 | 1 | bits[58:55] (the IGNORED field in the stage 1/2 block and page descriptors), res |
| E084 | figure | image_read | 1 | 0.08 | 0 | RES0 — bit[1] (Register bit[1] is RES0), per chunk arm_p2521_2610:171. |
| E090 | negative | any | 1 | 0.40 | 1 | The document does not contain the die area of the processor in square millimetre |
| E091 | negative | any | 1 | 0.10 | 0 | Arm (ARM Limited) — this is the ARM Architecture Reference Manual; the printed p |
| E092 | negative | any | 1 | 0.43 | 1 | The document does not provide a typical power consumption value in milliwatts. |