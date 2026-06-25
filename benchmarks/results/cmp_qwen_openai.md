# BENCHMARK — 독립 골드셋 21문항 (Reader provider = openai)

> 문서: ARMv8-Reference-Manual (expanded parse, data/ir_full). 검색 Recall(출처 검색율) + Reader 최종 답변 F1/EM(SQuAD식).
> 이 셋은 모델-비의존(gold=원문) → **Reader LLM을 바꿔 같은 셋으로 비교**할 수 있다.


## 집계

| 지표 | 값 |
|------|----|
| 검색 Recall (gold 출처 검색율) | **100.00%** |
| 답변 F1 (토큰, gold max) | **0.514** |
| 답변 EM (완전/포함 일치) | **38.10%** |
| 평균 LLM 호출 수 (루프) | **5.62** |
| 평균 답변 시간 (초) | **5.99** |


## 카테고리별 (컨텍스트: 표/그림/텍스트)

| category | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| concept | 3 | 100% | 0.55 | 33% |
| figure | 3 | 100% | 0.26 | 0% |
| figure-loc | 3 | 100% | 0.70 | 67% |
| heading-loc | 3 | 100% | 1.00 | 100% |
| negative | 3 | 100% | 0.25 | 67% |
| table | 3 | 100% | 0.51 | 0% |
| table-loc | 3 | 100% | 0.33 | 0% |

## 도구별 (의도 도구 커버리지)

| tool | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| any | 3 | 100% | 0.25 | 67% |
| image_read | 3 | 100% | 0.26 | 0% |
| keyword_search | 5 | 100% | 0.41 | 0% |
| page_index_search | 6 | 100% | 0.85 | 83% |
| read_chunk | 1 | 100% | 0.50 | 0% |
| semantic_search | 3 | 100% | 0.55 | 33% |

## 문항별

| ID | cat | tool | Recall | F1 | EM | 답변(발췌) |
|----|----|----|:-:|:-:|:-:|----|
| E001 | figure-loc | page_index_search | 1 | 1.00 | 1 | D1-1899 |
| E002 | figure-loc | page_index_search | 1 | 1.00 | 1 | D4-2061 |
| E003 | figure-loc | page_index_search | 1 | 0.09 | 0 | The figure/diagram in the section about 'D7 AArch64 System Register Descriptions |
| E016 | table-loc | keyword_search | 1 | 0.00 | 0 | E2-1658 |
| E017 | table-loc | keyword_search | 1 | 0.50 | 0 | D7-2430 |
| E018 | table-loc | keyword_search | 1 | 0.50 | 0 | D7-2610 |
| E024 | heading-loc | page_index_search | 1 | 1.00 | 1 | D1-1891 |
| E025 | heading-loc | page_index_search | 1 | 1.00 | 1 | D3-1991 |
| E026 | heading-loc | page_index_search | 1 | 1.00 | 1 | D4-2090 |
| E040 | concept | semantic_search | 1 | 0.60 | 0 | Interprocessing is covered in section D1.20. |
| E041 | concept | semantic_search | 1 | 0.88 | 1 | The key term is: "The access size is smaller or larger than the address region b |
| E042 | concept | semantic_search | 1 | 0.17 | 0 | Synchronous external abort, not on translation table walk |
| E065 | table | keyword_search | 1 | 0.61 | 0 | HIFAR is an alias of the Secure IFAR when EL3 uses AArch32 because of the aliasi |
| E066 | table | read_chunk | 1 | 0.50 | 0 | Non-shareable |
| E067 | table | keyword_search | 1 | 0.43 | 0 | Overflow exception has occurred |
| E082 | figure | image_read | 1 | 0.60 | 0 | The Secure monitor operates at Exception Level EL3. |
| E083 | figure | image_read | 1 | 0.00 | 0 | bits[58:55] |
| E084 | figure | image_read | 1 | 0.18 | 0 | Bits labeled as RES0 are bits [31:8], bit [5], and bit [0]. |
| E090 | negative | any | 1 | 0.35 | 1 | The document does not contain information about the die area of the processor in |
| E091 | negative | any | 1 | 0.00 | 0 | ARM |
| E092 | negative | any | 1 | 0.40 | 1 | The document does not provide a specific typical power consumption value in mill |