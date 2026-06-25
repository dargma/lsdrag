# LONG-CONTEXT BENCHMARK — 20문항 (Reader=openai)

> ARMv8 매뉴얼의 '파싱된 일부'(~1800p, 20조각, 전체 6354p 아님) — data/ir_full
> 여러 출처 통합/집계/참조추적 측정 — 단답 grounding과 별개.


## 집계

| 지표 | 값 |
|----|----|
| multi-source recall (필요 출처 중 검색됨) | **72.50%** |
| all-sources rate (필요 출처 전부 검색) | **55.00%** |
| synthesis (집계 커버리지 / token F1) | **0.236** |
| 평균 LLM 호출 | **6.25** |  평균 시간 | **8.48s** |


## 카테고리별

| category | n | multi-src recall | all-src | synthesis |
|----|:-:|:-:|:-:|:-:|
| aggregation | 8 | 100% | 100% | 0.34 |
| crossref | 6 | 67% | 33% | 0.04 |
| multihop | 6 | 42% | 17% | 0.29 |

## 문항별

| ID | cat | multi-src | all-src | synthesis | 답변(발췌) |
|----|----|:-:|:-:|:-:|----|
| L01 | aggregation | 1.00 | 1 | 0.00 | The page D1-1895 containing the table on "There are a small number of  |
| L02 | aggregation | 1.00 | 1 | 0.67 | The table on page D1-1896 (section 'D1.20.2 State of the general-purpo |
| L03 | aggregation | 1.00 | 1 | 0.00 | The page D1-1897 containing the table in section 'D1 The AArch64 Syste |
| L04 | aggregation | 1.00 | 1 | 0.33 | The table on page D1-1897 titled "Table D1-88 SPSR and ELR mappings be |
| L05 | aggregation | 1.00 | 1 | 0.50 | The table on page D1-1900 in section D1.21.2 "Support for Exception le |
| L06 | aggregation | 1.00 | 1 | 0.33 | The table on page D2-1908 under the section "Software Step exceptions" |
| L07 | aggregation | 1.00 | 1 | 0.25 | The table on page D2-1909 in section "D2 AArch64 Self-hosted Debug D2. |
| L08 | aggregation | 1.00 | 1 | 0.67 | The table on page D2-1912 in section 'D2 AArch64 Self-hosted Debug D2. |
| L09 | crossref | 0.50 | 0 | 0.05 | The content on page D1-1896 about 'D1.20.2 State of the general-purpos |
| L10 | crossref | 0.50 | 0 | 0.04 | The content on page D1-1897 about 'D1 The AArch64 System Level Program |
| L11 | crossref | 1.00 | 1 | 0.06 | The content on page D1-1899 about 'D1.21 The effect of implementation  |
| L12 | crossref | 0.50 | 0 | 0.03 | The content on page D1-1899 about 'D1.21.1 Implication of Exception le |
| L13 | crossref | 1.00 | 1 | 0.04 | The content on page D1-1900 about "D1 The AArch64 System Level Program |
| L14 | crossref | 0.50 | 0 | 0.03 | The content on page D1-1900 about "D1 The AArch64 System Level Program |
| L15 | multihop | 0.50 | 0 | 0.27 | When a hardware watchpoint is programmed for a data address range and  |
| L16 | multihop | 0.50 | 0 | 0.27 | The preferred return address for a Software Step exception is the addr |
| L17 | multihop | 0.50 | 0 | 0.21 | A debugger can configure watchpoints and breakpoints so that a Breakpo |
| L18 | multihop | 0.00 | 0 | 0.49 | When the implementation includes ARMv8.1-VHE and EL2 uses AArch64 with |
| L19 | multihop | 1.00 | 1 | 0.26 | The specific chunks for pages D2-1926 and D2-1927 were not found in th |
| L20 | multihop | 0.00 | 0 | 0.23 | The PE records information about a Breakpoint exception in the ESR_ELx |