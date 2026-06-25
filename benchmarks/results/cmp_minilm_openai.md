# BENCHMARK — 독립 골드셋 21문항 (Reader provider = openai)

> 문서: ARMv8-Reference-Manual (expanded parse, data/ir_full). 검색 Recall(출처 검색율) + Reader 최종 답변 F1/EM(SQuAD식).
> 이 셋은 모델-비의존(gold=원문) → **Reader LLM을 바꿔 같은 셋으로 비교**할 수 있다.


## 집계

| 지표 | 값 |
|------|----|
| 검색 Recall (gold 출처 검색율) | **100.00%** |
| 답변 F1 (토큰, gold max) | **0.494** |
| 답변 EM (완전/포함 일치) | **38.10%** |
| 평균 LLM 호출 수 (루프) | **5.05** |
| 평균 답변 시간 (초) | **5.45** |


## 카테고리별 (컨텍스트: 표/그림/텍스트)

| category | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| concept | 3 | 100% | 0.35 | 0% |
| figure | 3 | 100% | 0.23 | 0% |
| figure-loc | 3 | 100% | 0.83 | 67% |
| heading-loc | 3 | 100% | 1.00 | 100% |
| negative | 3 | 100% | 0.25 | 67% |
| table | 3 | 100% | 0.75 | 33% |
| table-loc | 3 | 100% | 0.04 | 0% |

## 도구별 (의도 도구 커버리지)

| tool | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| any | 3 | 100% | 0.25 | 67% |
| image_read | 3 | 100% | 0.23 | 0% |
| keyword_search | 5 | 100% | 0.28 | 0% |
| page_index_search | 6 | 100% | 0.92 | 83% |
| read_chunk | 1 | 100% | 1.00 | 100% |
| semantic_search | 3 | 100% | 0.35 | 0% |

## 문항별

| ID | cat | tool | Recall | F1 | EM | 답변(발췌) |
|----|----|----|:-:|:-:|:-:|----|
| E001 | figure-loc | page_index_search | 1 | 1.00 | 1 | D1-1899 |
| E002 | figure-loc | page_index_search | 1 | 1.00 | 1 | D4-2061 |
| E003 | figure-loc | page_index_search | 1 | 0.50 | 0 | D7-2201 |
| E016 | table-loc | keyword_search | 1 | 0.00 | 0 | arm_p2071_2160:1658 |
| E017 | table-loc | keyword_search | 1 | 0.00 | 0 | arm_p2341_2430:754 is the chunk containing the table mentioning 'Support for FCS |
| E018 | table-loc | keyword_search | 1 | 0.12 | 0 | The table mentioning 'Implemented and disabled. ExternalSecureInvasiveDebugEnabl |
| E024 | heading-loc | page_index_search | 1 | 1.00 | 1 | D1-1891 |
| E025 | heading-loc | page_index_search | 1 | 1.00 | 1 | D3-1991 |
| E026 | heading-loc | page_index_search | 1 | 1.00 | 1 | D4-2090 |
| E040 | concept | semantic_search | 1 | 0.60 | 0 | Interprocessing is covered in section D1.20. |
| E041 | concept | semantic_search | 1 | 0.29 | 0 | Access size mismatch |
| E042 | concept | semantic_search | 1 | 0.17 | 0 | Synchronous external abort, not on translation table walk |
| E065 | table | keyword_search | 1 | 0.83 | 0 | HIFAR is an alias of the Secure IFAR when EL3 uses AArch32 because EL3 is using  |
| E066 | table | read_chunk | 1 | 1.00 | 1 | Outer Shareable |
| E067 | table | keyword_search | 1 | 0.43 | 0 | Overflow exception has occurred |
| E082 | figure | image_read | 1 | 0.50 | 0 | EL3 |
| E083 | figure | image_read | 1 | 0.00 | 0 | Bits[58:55] |
| E084 | figure | image_read | 1 | 0.18 | 0 | Bits labeled as RES0 are bits [31:8], bit [5], and bit [0]. |
| E090 | negative | any | 1 | 0.35 | 1 | The document does not contain information about the die area of the processor in |
| E091 | negative | any | 1 | 0.00 | 0 | ARM |
| E092 | negative | any | 1 | 0.40 | 1 | The document does not provide a specific typical power consumption value in mill |