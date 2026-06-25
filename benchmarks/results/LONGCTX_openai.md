# LONG-CONTEXT BENCHMARK — 20문항 (Reader=openai)

> ARMv8 매뉴얼의 '파싱된 일부'(~1800p, 20조각, 전체 6354p 아님) — data/ir_full
> 여러 출처 통합/집계/참조추적 측정 — 단답 grounding과 별개.


## 집계

| 지표 | 값 |
|----|----|
| multi-source recall (필요 출처 중 검색됨) | **80.00%** |
| all-sources rate (필요 출처 전부 검색) | **65.00%** |
| synthesis (집계 커버리지 / token F1) | **0.271** |
| 평균 LLM 호출 | **6.50** |  평균 시간 | **11.42s** |


## 카테고리별

| category | n | multi-src recall | all-src | synthesis |
|----|:-:|:-:|:-:|:-:|
| aggregation | 8 | 100% | 100% | 0.42 |
| crossref | 6 | 83% | 67% | 0.04 |
| multihop | 6 | 50% | 17% | 0.30 |

## 문항별

| ID | cat | multi-src | all-src | synthesis | 답변(발췌) |
|----|----|:-:|:-:|:-:|----|
| L01 | aggregation | 1.00 | 1 | 1.00 | The page D1-1895 contains two tables related to AArch32 System registe |
| L02 | aggregation | 1.00 | 1 | 0.00 | The page D1-1896 with the section 'D1.20.2 State of the general-purpos |
| L03 | aggregation | 1.00 | 1 | 0.00 | The table on page D1-1897 titled "Table D1-87 General-purpose register |
| L04 | aggregation | 1.00 | 1 | 1.00 | The table on page D1-1897 in section 'D1.20.3 SPSR, ELR, and AArch64 S |
| L05 | aggregation | 1.00 | 1 | 0.00 | The table titled "Table D1-89 Supported combinations of Exception leve |
| L06 | aggregation | 1.00 | 1 | 0.33 | The table titled "Software Step exceptions" on page D2-1908 contains 3 |
| L07 | aggregation | 1.00 | 1 | 1.00 | The table on page D2-1909 titled "Table D2-1 PE behavior and pseudocod |
| L08 | aggregation | 1.00 | 1 | 0.00 | The table on page D2-1912 in section 'D2 AArch64 Self-hosted Debug D2. |
| L09 | crossref | 0.50 | 0 | 0.03 | The content on page D1-1896 about "D1.20.2 State of the general-purpos |
| L10 | crossref | 0.50 | 0 | 0.02 | The content on page D1-1897 about "D1 The AArch64 System Level Program |
| L11 | crossref | 1.00 | 1 | 0.05 | The content on page D1-1899 about 'D1.21 The effect of implementation  |
| L12 | crossref | 1.00 | 1 | 0.11 | The content on page D1-1899 about 'D1.21.1 Implication of Exception le |
| L13 | crossref | 1.00 | 1 | 0.03 | The content on page D1-1900 about 'D1 The AArch64 System Level Program |
| L14 | crossref | 1.00 | 1 | 0.03 | The content on page D1-1900 about "D1 The AArch64 System Level Program |
| L15 | multihop | 0.50 | 0 | 0.38 | When a hardware watchpoint is programmed for a data address range and  |
| L16 | multihop | 0.50 | 0 | 0.18 | The preferred return address for a Software Step exception is the addr |
| L17 | multihop | 0.50 | 0 | 0.14 | To configure a debugger so that a Breakpoint debug event is generated  |
| L18 | multihop | 0.00 | 0 | 0.47 | When the implementation includes ARMv8.1-VHE and EL2 uses AArch64 with |
| L19 | multihop | 1.00 | 1 | 0.21 | The breakpoint can be triggered under conditions involving data addres |
| L20 | multihop | 0.50 | 0 | 0.42 | The PE (Processing Element) records information about a Breakpoint exc |