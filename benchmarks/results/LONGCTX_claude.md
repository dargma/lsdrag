# LONG-CONTEXT BENCHMARK — 20문항 (Reader=claude_code)

> ARMv8 매뉴얼의 '파싱된 일부'(~1800p, 20조각, 전체 6354p 아님) — data/ir_full
> 여러 출처 통합/집계/참조추적 측정 — 단답 grounding과 별개.


## 집계

| 지표 | 값 |
|----|----|
| multi-source recall (필요 출처 중 검색됨) | **70.00%** |
| all-sources rate (필요 출처 전부 검색) | **65.00%** |
| synthesis (집계 커버리지 / token F1) | **0.411** |
| 평균 LLM 호출 | **3.80** |  평균 시간 | **29.09s** |


## 카테고리별

| category | n | multi-src recall | all-src | synthesis |
|----|:-:|:-:|:-:|:-:|
| aggregation | 8 | 100% | 100% | 0.67 |
| crossref | 6 | 92% | 83% | 0.04 |
| multihop | 6 | 8% | 0% | 0.44 |

## 문항별

| ID | cat | multi-src | all-src | synthesis | 답변(발췌) |
|----|----|:-:|:-:|:-:|----|
| L01 | aggregation | 1.00 | 1 | 1.00 | Table D1-84 on page D1-1895 ('There are a small number of AArch32 Syst |
| L02 | aggregation | 1.00 | 1 | 1.00 | On printed page **D1-1896**, section *D1.20.2 State of the general-pur |
| L03 | aggregation | 1.00 | 1 | 1.00 | On page **D1-1897** (section 'D1 The AArch64 System Level Programmers' |
| L04 | aggregation | 1.00 | 1 | 1.00 | Table D1-88 "SPSR and ELR mappings between AArch32 state and AArch64 s |
| L05 | aggregation | 1.00 | 1 | 0.00 | I have the table location but the printed table rows are truncated in  |
| L06 | aggregation | 1.00 | 1 | 0.33 | On page D2-1908 (section 'Software Step exceptions'), the table lists  |
| L07 | aggregation | 1.00 | 1 | 0.00 | Error: Maximum loops exceeded and failed to generate final answer. |
| L08 | aggregation | 1.00 | 1 | 1.00 | On page D2-1912 (section 'D2 AArch64 Self-hosted Debug, D2.3 Routing d |
| L09 | crossref | 1.00 | 1 | 0.00 | Error: Maximum loops exceeded and failed to generate final answer. |
| L10 | crossref | 0.50 | 0 | 0.06 | Page D1-1897 (section 'D1.20 Interprocessing', Table D1-87 on general- |
| L11 | crossref | 1.00 | 1 | 0.04 | Page D1-1899 (section 'D1.21 The effect of implementation choices on t |
| L12 | crossref | 1.00 | 1 | 0.08 | Page D1-1899 (section 'D1.21.1 Implication of Exception levels impleme |
| L13 | crossref | 1.00 | 1 | 0.04 | Page D1-1900 (in 'D1 The AArch64 System Level Programmers’ Model', sec |
| L14 | crossref | 1.00 | 1 | 0.05 | On page D1-1900 (under 'D1 The AArch64 System Level Programmers’ Model |
| L15 | multihop | 0.00 | 0 | 0.37 | When a hardware watchpoint is programmed for a data address range, if  |
| L16 | multihop | 0.00 | 0 | 0.23 | For a Software Step exception, the preferred return address is the add |
| L17 | multihop | 0.00 | 0 | 0.22 | On page D2-1926 of the ARM manual (arm_p1891_1980), the document descr |
| L18 | multihop | 0.00 | 0 | 0.54 | When the implementation includes ARMv8.1-VHE and EL2 uses AArch64 with |
| L19 | multihop | 0.50 | 0 | 0.60 | Across pages D2-1926–D2-1927, this breakpoint can be triggered by the  |
| L20 | multihop | 0.00 | 0 | 0.65 | On taking a Breakpoint exception, the PE records information about the |