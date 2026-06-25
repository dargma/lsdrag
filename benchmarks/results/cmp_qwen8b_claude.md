# BENCHMARK — 독립 골드셋 21문항 (Reader provider = claude_code)

> 문서: ARMv8-Reference-Manual (expanded parse, data/ir_full). 검색 Recall(출처 검색율) + Reader 최종 답변 F1/EM(SQuAD식).
> 이 셋은 모델-비의존(gold=원문) → **Reader LLM을 바꿔 같은 셋으로 비교**할 수 있다.


## 집계

| 지표 | 값 |
|------|----|
| 검색 Recall (gold 출처 검색율) | **100.00%** |
| 답변 F1 (토큰, gold max) | **0.482** |
| 답변 EM (완전/포함 일치) | **57.14%** |
| 평균 LLM 호출 수 (루프) | **2.71** |
| 평균 답변 시간 (초) | **16.54** |


## 카테고리별 (컨텍스트: 표/그림/텍스트)

| category | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| concept | 3 | 100% | 0.50 | 67% |
| figure | 3 | 100% | 0.38 | 33% |
| figure-loc | 3 | 100% | 0.83 | 67% |
| heading-loc | 3 | 100% | 1.00 | 100% |
| negative | 3 | 100% | 0.20 | 100% |
| table | 3 | 100% | 0.29 | 33% |
| table-loc | 3 | 100% | 0.17 | 0% |

## 도구별 (의도 도구 커버리지)

| tool | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| any | 3 | 100% | 0.20 | 100% |
| image_read | 3 | 100% | 0.38 | 33% |
| keyword_search | 5 | 100% | 0.26 | 20% |
| page_index_search | 6 | 100% | 0.92 | 83% |
| read_chunk | 1 | 100% | 0.05 | 0% |
| semantic_search | 3 | 100% | 0.50 | 67% |

## 문항별

| ID | cat | tool | Recall | F1 | EM | 답변(발췌) |
|----|----|----|:-:|:-:|:-:|----|
| E001 | figure-loc | page_index_search | 1 | 1.00 | 1 | D1-1899 |
| E002 | figure-loc | page_index_search | 1 | 1.00 | 1 | D4-2061 |
| E003 | figure-loc | page_index_search | 1 | 0.50 | 0 | D7-2198 |
| E016 | table-loc | keyword_search | 1 | 0.00 | 0 | p2071 |
| E017 | table-loc | keyword_search | 1 | 0.00 | 0 | Page label E2-2804 is not present; the table mentioning 'Support for FCSE.' is i |
| E018 | table-loc | keyword_search | 1 | 0.50 | 0 | D7-2556 |
| E024 | heading-loc | page_index_search | 1 | 1.00 | 1 | D1-1891 |
| E025 | heading-loc | page_index_search | 1 | 1.00 | 1 | D3-1991 |
| E026 | heading-loc | page_index_search | 1 | 1.00 | 1 | D4-2090 |
| E040 | concept | semantic_search | 1 | 1.00 | 1 | D1.20 Interprocessing |
| E041 | concept | semantic_search | 1 | 0.16 | 0 | Address size fault (page E2-2804... ) — let me cite the actual printed page labe |
| E042 | concept | semantic_search | 1 | 0.33 | 1 | A synchronous external abort on an instruction fetch is taken precisely using th |
| E065 | table | keyword_search | 1 | 0.49 | 0 | HIFAR is an alias of the Secure IFAR (when EL3 uses AArch32). Key term: Secure I |
| E066 | table | read_chunk | 1 | 0.05 | 0 | Non-shareable. Per Table D4-44 (Combining the stage 1 and stage 2 Shareability a |
| E067 | table | keyword_search | 1 | 0.33 | 1 | {"final": "Set to 1 indicates: Overflow floating-point exception occurred during |
| E082 | figure | image_read | 1 | 0.50 | 0 | EL3 |
| E083 | figure | image_read | 1 | 0.50 | 1 | bits[58:55] (reserved for software use), printed page label E1-2080 |
| E084 | figure | image_read | 1 | 0.14 | 0 | Reserved bits — labeled RES0. |
| E090 | negative | any | 1 | 0.22 | 1 | The manual does not contain a die area specification (in mm²) for the processor  |
| E091 | negative | any | 1 | 0.21 | 1 | The document does not state the chip manufacturer; it is an Arm processor specif |
| E092 | negative | any | 1 | 0.18 | 1 | Not specified — this manual gives no typical power consumption value in milliwat |