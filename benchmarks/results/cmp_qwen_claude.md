# BENCHMARK — 독립 골드셋 21문항 (Reader provider = claude_code)

> 문서: ARMv8-Reference-Manual (expanded parse, data/ir_full). 검색 Recall(출처 검색율) + Reader 최종 답변 F1/EM(SQuAD식).
> 이 셋은 모델-비의존(gold=원문) → **Reader LLM을 바꿔 같은 셋으로 비교**할 수 있다.


## 집계

| 지표 | 값 |
|------|----|
| 검색 Recall (gold 출처 검색율) | **100.00%** |
| 답변 F1 (토큰, gold max) | **0.525** |
| 답변 EM (완전/포함 일치) | **61.90%** |
| 평균 LLM 호출 수 (루프) | **2.81** |
| 평균 답변 시간 (초) | **17.73** |


## 카테고리별 (컨텍스트: 표/그림/텍스트)

| category | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| concept | 3 | 100% | 0.24 | 33% |
| figure | 3 | 100% | 0.37 | 67% |
| figure-loc | 3 | 100% | 0.50 | 33% |
| heading-loc | 3 | 100% | 1.00 | 100% |
| negative | 3 | 100% | 0.24 | 67% |
| table | 3 | 100% | 0.66 | 67% |
| table-loc | 3 | 100% | 0.67 | 67% |

## 도구별 (의도 도구 커버리지)

| tool | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| any | 3 | 100% | 0.24 | 67% |
| image_read | 3 | 100% | 0.37 | 67% |
| keyword_search | 5 | 100% | 0.60 | 60% |
| page_index_search | 6 | 100% | 0.75 | 67% |
| read_chunk | 1 | 100% | 1.00 | 100% |
| semantic_search | 3 | 100% | 0.24 | 33% |

## 문항별

| ID | cat | tool | Recall | F1 | EM | 답변(발췌) |
|----|----|----|:-:|:-:|:-:|----|
| E001 | figure-loc | page_index_search | 1 | 1.00 | 1 | D1-1899 |
| E002 | figure-loc | page_index_search | 1 | 0.00 | 0 | arm_p1981_2070:1588 |
| E003 | figure-loc | page_index_search | 1 | 0.50 | 0 | D7-2198 |
| E016 | table-loc | keyword_search | 1 | 1.00 | 1 | D5-2150 |
| E017 | table-loc | keyword_search | 1 | 1.00 | 1 | D7-2372 |
| E018 | table-loc | keyword_search | 1 | 0.00 | 0 | {"tool": "page_index_search", "args": {"doc_id": "arm_p2521_2610", "heading": "E |
| E024 | heading-loc | page_index_search | 1 | 1.00 | 1 | D1-1891 |
| E025 | heading-loc | page_index_search | 1 | 1.00 | 1 | D3-1991 |
| E026 | heading-loc | page_index_search | 1 | 1.00 | 1 | D4-2090 |
| E040 | concept | semantic_search | 1 | 0.35 | 1 | D1.20 Interprocessing (chunk arm_p1891_1980; no printed page label available in  |
| E041 | concept | semantic_search | 1 | 0.00 | 0 | CONSTRAINED UNPREDICTABLE |
| E042 | concept | semantic_search | 1 | 0.35 | 0 | Synchronous external abort on an instruction fetch is taken precisely as the Ins |
| E065 | table | keyword_search | 1 | 0.07 | 0 | HIFAR (page label arm_p1891_1980 — printed page label not shown in results) |
| E066 | table | read_chunk | 1 | 1.00 | 1 | Outer Shareable |
| E067 | table | keyword_search | 1 | 0.91 | 1 | 1 indicates an Overflow floating-point exception occurred during execution of th |
| E082 | figure | image_read | 1 | 0.50 | 1 | EL3. The Secure monitor operates at Exception level 3 (EL3). |
| E083 | figure | image_read | 1 | 0.62 | 1 | IGNORED (the descriptor IGNORED bits — reserved for software use) |
| E084 | figure | image_read | 1 | 0.00 | 0 | Reserved |
| E090 | negative | any | 1 | 0.40 | 1 | The document does not state the processor's die area in square millimetres. |
| E091 | negative | any | 1 | 0.00 | 0 | ARM Limited |
| E092 | negative | any | 1 | 0.33 | 1 | Not stated — this manual does not give a typical power consumption figure in mil |