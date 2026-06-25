# BENCHMARK — 독립 골드셋 99문항 (Reader provider = openai)

> 문서: ARMv8-Reference-Manual (expanded parse, data/ir_full). 검색 Recall(출처 검색율) + Reader 최종 답변 F1/EM(SQuAD식).
> 이 셋은 모델-비의존(gold=원문) → **Reader LLM을 바꿔 같은 셋으로 비교**할 수 있다.


## 집계

| 지표 | 값 |
|------|----|
| 검색 Recall (gold 출처 검색율) | **92.93%** |
| 답변 F1 (토큰, gold max) | **0.386** |
| 답변 EM (완전/포함 일치) | **34.34%** |
| 평균 LLM 호출 수 (루프) | **4.66** |
| 평균 답변 시간 (초) | **5.22** |


## 카테고리별 (컨텍스트: 표/그림/텍스트)

| category | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| concept | 25 | 96% | 0.50 | 36% |
| figure | 8 | 75% | 0.29 | 50% |
| figure-loc | 15 | 100% | 0.53 | 47% |
| heading-loc | 16 | 100% | 0.46 | 25% |
| negative | 10 | 100% | 0.26 | 80% |
| table | 17 | 76% | 0.15 | 0% |
| table-loc | 8 | 100% | 0.38 | 25% |

## 도구별 (의도 도구 커버리지)

| tool | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| any | 10 | 100% | 0.26 | 80% |
| image_read | 8 | 75% | 0.29 | 50% |
| keyword_search | 17 | 88% | 0.27 | 12% |
| page_index_search | 31 | 100% | 0.49 | 35% |
| read_chunk | 8 | 75% | 0.13 | 0% |
| semantic_search | 25 | 96% | 0.50 | 36% |

## 문항별

| ID | cat | tool | Recall | F1 | EM | 답변(발췌) |
|----|----|----|:-:|:-:|:-:|----|
| E001 | figure-loc | page_index_search | 1 | 0.18 | 1 | The figure/diagram in the section about 'D1.21.1 Implication of Exception levels |
| E002 | figure-loc | page_index_search | 1 | 1.00 | 1 | D4-2061 |
| E003 | figure-loc | page_index_search | 1 | 0.50 | 0 | D7-2201 |
| E004 | figure-loc | page_index_search | 1 | 0.50 | 0 | D7-2442 |
| E005 | figure-loc | page_index_search | 1 | 1.00 | 1 | D7-2564 |
| E006 | figure-loc | page_index_search | 1 | 0.00 | 0 | No figure/diagram is explicitly indicated on the printed page label D4-2098, whi |
| E007 | figure-loc | page_index_search | 1 | 0.00 | 0 | F5-3668 |
| E008 | figure-loc | page_index_search | 1 | 1.00 | 1 | F3-2882 |
| E009 | figure-loc | page_index_search | 1 | 1.00 | 1 | F3-2916 |
| E010 | figure-loc | page_index_search | 1 | 1.00 | 1 | F4-2955 |
| E011 | figure-loc | page_index_search | 1 | 0.00 | 0 | The figure/diagram in the section about 'Decode for all variants of this encodin |
| E012 | figure-loc | page_index_search | 1 | 0.50 | 0 | F5-3103 |
| E013 | figure-loc | page_index_search | 1 | 0.11 | 0 | The figure/diagram in the section about 'F5.1.102 LSRS (immediate)' is on printe |
| E014 | figure-loc | page_index_search | 1 | 1.00 | 1 | F5-3333 |
| E015 | figure-loc | page_index_search | 1 | 0.12 | 0 | The figure/diagram in the section about 'SUBS{ }{ } { ,} SP, #' is on printed pa |
| E016 | table-loc | keyword_search | 1 | 1.00 | 1 | D5-2150 |
| E017 | table-loc | keyword_search | 1 | 1.00 | 1 | D7-2372 |
| E018 | table-loc | keyword_search | 1 | 0.00 | 0 | No page contains the table mentioning 'Implemented and disabled. ExternalSecureI |
| E019 | table-loc | keyword_search | 1 | 0.00 | 0 | D7-2421 |
| E020 | table-loc | keyword_search | 1 | 0.50 | 0 | F5-3033 |
| E021 | table-loc | keyword_search | 1 | 0.00 | 0 | F3-2888 |
| E022 | table-loc | keyword_search | 1 | 0.00 | 0 | E2-1056 |
| E023 | table-loc | keyword_search | 1 | 0.50 | 0 | F5-3584 |
| E024 | heading-loc | page_index_search | 1 | 1.00 | 1 | D1-1891 |
| E025 | heading-loc | page_index_search | 1 | 0.00 | 0 | E2-2801 |
| E026 | heading-loc | page_index_search | 1 | 0.13 | 0 | The section titled "Memory type and Cacheability" is on printed page label D4-20 |
| E027 | heading-loc | page_index_search | 1 | 0.50 | 0 | D6-2180 |
| E028 | heading-loc | page_index_search | 1 | 0.50 | 0 | D7-2270 |
| E029 | heading-loc | page_index_search | 1 | 0.50 | 0 | D7-2198 |
| E030 | heading-loc | page_index_search | 1 | 0.00 | 0 | 503 |
| E031 | heading-loc | page_index_search | 1 | 0.50 | 0 | D7-2562 |
| E032 | heading-loc | page_index_search | 1 | 1.00 | 1 | D7-2685 |
| E033 | heading-loc | page_index_search | 1 | 0.50 | 0 | E2-2791 |
| E034 | heading-loc | page_index_search | 1 | 1.00 | 1 | F3-2924 |
| E035 | heading-loc | page_index_search | 1 | 0.50 | 0 | F5-3003 |
| E036 | heading-loc | page_index_search | 1 | 0.00 | 0 | The section titled 'Decode for this encoding' is on printed page label F3-2898. |
| E037 | heading-loc | page_index_search | 1 | 1.00 | 1 | F5-3306 |
| E038 | heading-loc | page_index_search | 1 | 0.13 | 0 | The section titled "Operation for all encodings" is on printed page label F5-300 |
| E039 | heading-loc | page_index_search | 1 | 0.09 | 0 | The section titled 'F6 T32 and A32 Advanced SIMD and floating-point Instruction  |
| E040 | concept | semantic_search | 1 | 0.46 | 0 | Interprocessing is covered in section D1.20 in the AArch64 manual. |
| E041 | concept | semantic_search | 1 | 0.00 | 0 | Watchpoint exception |
| E042 | concept | semantic_search | 1 | 0.86 | 0 | Instruction Abort exception |
| E043 | concept | semantic_search | 1 | 0.12 | 0 | Benefit: Start translation at a higher level (next translation level) with fewer |
| E044 | concept | semantic_search | 1 | 0.63 | 0 | Inner Shareable and Write-Back attributes describe Normal memory. |
| E045 | concept | semantic_search | 1 | 0.14 | 0 | TLB lockdown is described as heavily dependent on the microarchitecture, making  |
| E046 | concept | semantic_search | 1 | 1.00 | 1 | ARM Limited or its affiliates |
| E047 | concept | semantic_search | 1 | 0.83 | 1 | This control does not cause any instructions to be trapped when set to 1. |
| E048 | concept | semantic_search | 1 | 0.11 | 0 | FAR_EL1 holds the address of the faulting instruction for a memory fault from a  |
| E049 | concept | semantic_search | 1 | 0.38 | 0 | Trapped to EL2 |
| E050 | concept | semantic_search | 1 | 0.29 | 0 | The statement applies when EL3 is the highest Exception level implemented. |
| E051 | concept | semantic_search | 1 | 0.00 | 0 | 0, 1 |
| E052 | concept | semantic_search | 1 | 0.50 | 1 | VBAR_EL2[31:0] is architecturally mapped to the AArch32 system register HVBAR. |
| E053 | concept | semantic_search | 1 | 0.48 | 1 | ARM strongly recommends that bits ROMADDR[(PAsize-1):32] are zero in any system  |
| E054 | concept | semantic_search | 1 | 0.62 | 0 | Clock frequency indicates the system counter clock frequency. |
| E055 | concept | semantic_search | 1 | 0.18 | 0 | At EL0, the accessible PSTATE fields are:

- Condition flags N, Z, C, V (bits [3 |
| E056 | concept | semantic_search | 1 | 0.67 | 1 | The BigEndian() pseudocode function determines the current endianness of the dat |
| E057 | concept | semantic_search | 1 | 0.88 | 0 | Section F1.13 covers Advanced SIMD data-processing instructions. |
| E058 | concept | semantic_search | 1 | 1.00 | 1 | ARM Limited or its affiliates |
| E059 | concept | semantic_search | 1 | 0.62 | 1 | Alphabetical list of T32 and A32 base instructions is in section "F5.1 Alphabeti |
| E060 | concept | semantic_search | 1 | 0.44 | 1 | LDC pseudocode might trap to Hyp mode. |
| E061 | concept | semantic_search | 1 | 1.00 | 1 | ARM Limited or its affiliates |
| E062 | concept | semantic_search | 1 | 0.67 | 0 | ORR (T1 variant) is used inside an IT block. |
| E063 | concept | semantic_search | 1 | 0.59 | 0 | Alphabetical list of T32 and A32 instructions is in section "F5 Alphabetical lis |
| E064 | concept | semantic_search | 0 | 0.00 | 0 | The preferred syntax for the Empty Descending stack instruction is "LDMIB". |
| E065 | table | keyword_search | 1 | 0.69 | 0 | HIFAR is an alias of the Secure IFAR when EL3 uses AArch32 because of the "Secur |
| E066 | table | read_chunk | 1 | 0.50 | 0 | Non-shareable |
| E067 | table | keyword_search | 1 | 0.43 | 0 | Overflow exception has occurred. |
| E068 | table | read_chunk | 1 | 0.18 | 0 | The value 0001 adds the IT instructions to the ARM instruction set. |
| E069 | table | keyword_search | 0 | 0.00 | 0 | Trapped to EL2 |
| E070 | table | read_chunk | 1 | 0.33 | 0 | Normal memory |
| E071 | table | keyword_search | 1 | 0.00 | 0 | At EL1, when E2H is 1, TGE is 0, and NS is 1, the accessibility of the control i |
| E072 | table | read_chunk | 1 | 0.00 | 0 | The value of op1 for CNTP_CVAL_EL0 is 2. |
| E073 | table | keyword_search | 1 | 0.00 | 0 | The instruction corresponding to decode field L value 0 is "PUSH". |
| E074 | table | read_chunk | 0 | 0.00 | 0 | Floating-point move immediate |
| E075 | table | keyword_search | 0 | 0.00 | 0 | The 'cond' field bit pattern is bits [23:20]. For exceptions taken from AArch64, |
| E076 | table | read_chunk | 1 | 0.00 | 0 | Branching |
| E077 | table | keyword_search | 1 | 0.40 | 0 | Rn: bits 19-16
Rt: bits 15-12 |
| E078 | table | read_chunk | 0 | 0.00 | 0 | The document does not contain the bit range for the Rt field in the table. |
| E079 | table | keyword_search | 1 | 0.00 | 0 | imm5 field bit range: bits 11 to 7 |
| E080 | table | read_chunk | 1 | 0.00 | 0 | No information about the bit pattern for bits 15 to 11 in a table is found in th |
| E081 | table | keyword_search | 1 | 0.00 | 0 | base register |
| E082 | figure | image_read | 1 | 0.55 | 0 | The Secure monitor operates at Exception Level 3 (EL3). |
| E083 | figure | image_read | 1 | 0.73 | 1 | Bits[58:55] reserved for software use |
| E084 | figure | image_read | 1 | 0.09 | 0 | Bits [31:17] and [15:13] in the IFSR32_EL2 register are labeled as RES0. |
| E085 | figure | image_read | 1 | 0.50 | 1 | WXN bit [19] |
| E086 | figure | image_read | 1 | 0.50 | 1 | TXfull, bit [29] |
| E087 | figure | image_read | 0 | 0.00 | 0 | The document does not contain the term "item_01." Please provide more context or |
| E088 | figure | image_read | 1 | 0.00 | 1 | The only chunk found with "item_01" and "1211" keywords is the same chunk pg:108 |
| E089 | figure | image_read | 0 | 0.00 | 0 | imm32 |
| E090 | negative | any | 1 | 0.35 | 1 | The document does not contain information about the die area of the processor in |
| E091 | negative | any | 1 | 0.00 | 0 | ARM Limited |
| E092 | negative | any | 1 | 0.00 | 0 | By 500 at 20kHz, when the system lowers the clock frequency, to reduce power con |
| E093 | negative | any | 1 | 0.31 | 1 | The document does not contain information about the pinout of the JTAG debug con |
| E094 | negative | any | 1 | 0.27 | 1 | The document does not specify which physical GPIO pins are used for the debug UA |
| E095 | negative | any | 1 | 0.17 | 1 | The document does not specify a maximum clock frequency in MHz for the part. It  |
| E096 | negative | any | 1 | 0.33 | 1 | The document does not contain information about the unit price of the processor. |
| E097 | negative | any | 1 | 0.38 | 1 | The document does not contain information about the operating temperature range  |
| E098 | negative | any | 1 | 0.29 | 1 | The document does not contain information about the package type (BGA/QFP) of th |
| E099 | negative | any | 1 | 0.50 | 1 | The document does not contain information about the number of transistors in thi |