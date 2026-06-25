# BENCHMARK — 독립 골드셋 99문항 (Reader provider = openai)

> 문서: ARMv8-Reference-Manual (expanded parse, data/ir_full). 검색 Recall(출처 검색율) + Reader 최종 답변 F1/EM(SQuAD식).
> 이 셋은 모델-비의존(gold=원문) → **Reader LLM을 바꿔 같은 셋으로 비교**할 수 있다.


## 집계

| 지표 | 값 |
|------|----|
| 검색 Recall (gold 출처 검색율) | **95.96%** |
| 답변 F1 (토큰, gold max) | **0.463** |
| 답변 EM (완전/포함 일치) | **38.38%** |


## 카테고리별 (컨텍스트: 표/그림/텍스트)

| category | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| concept | 25 | 100% | 0.48 | 24% |
| figure | 8 | 88% | 0.36 | 38% |
| figure-loc | 15 | 100% | 0.79 | 73% |
| heading-loc | 16 | 100% | 0.68 | 44% |
| negative | 10 | 100% | 0.29 | 80% |
| table | 17 | 88% | 0.30 | 18% |
| table-loc | 8 | 88% | 0.06 | 0% |

## 도구별 (의도 도구 커버리지)

| tool | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| any | 10 | 100% | 0.29 | 80% |
| image_read | 8 | 88% | 0.36 | 38% |
| keyword_search | 17 | 88% | 0.19 | 6% |
| page_index_search | 31 | 100% | 0.73 | 58% |
| read_chunk | 8 | 88% | 0.29 | 25% |
| semantic_search | 25 | 100% | 0.48 | 24% |

## 문항별

| ID | cat | tool | Recall | F1 | EM | 답변(발췌) |
|----|----|----|:-:|:-:|:-:|----|
| E001 | figure-loc | page_index_search | 1 | 1.00 | 1 | D1-1899 |
| E002 | figure-loc | page_index_search | 1 | 1.00 | 1 | D4-2061 |
| E003 | figure-loc | page_index_search | 1 | 0.50 | 0 | D7-2198 |
| E004 | figure-loc | page_index_search | 1 | 0.50 | 0 | D7-2199 |
| E005 | figure-loc | page_index_search | 1 | 1.00 | 1 | D7-2564 |
| E006 | figure-loc | page_index_search | 1 | 1.00 | 1 | D7-2676 |
| E007 | figure-loc | page_index_search | 1 | 1.00 | 1 | E2-2804 |
| E008 | figure-loc | page_index_search | 1 | 1.00 | 1 | F3-2882 |
| E009 | figure-loc | page_index_search | 1 | 1.00 | 1 | F3-2916 |
| E010 | figure-loc | page_index_search | 1 | 1.00 | 1 | F4-2955 |
| E011 | figure-loc | page_index_search | 1 | 0.50 | 0 | F5-3006 |
| E012 | figure-loc | page_index_search | 1 | 0.00 | 0 | D7-2331 |
| E013 | figure-loc | page_index_search | 1 | 1.00 | 1 | F5-3245 |
| E014 | figure-loc | page_index_search | 1 | 1.00 | 1 | F5-3333 |
| E015 | figure-loc | page_index_search | 1 | 0.29 | 1 | F5-3561, F5-3570, F5-3564, F5-3572, F5-3560, F5-3569 |
| E016 | table-loc | keyword_search | 1 | 0.00 | 0 | arm_p2071_2160:1658 |
| E017 | table-loc | keyword_search | 1 | 0.00 | 0 | arm_p2341_2430:754 (no printed page label found) |
| E018 | table-loc | keyword_search | 0 | 0.00 | 0 | The exact phrase "Implemented and disabled. ExternalSecureInvasiveDebugEnabled() |
| E019 | table-loc | keyword_search | 1 | 0.00 | 0 | D7-2590 |
| E020 | table-loc | keyword_search | 1 | 0.00 | 0 | F4-2971 |
| E021 | table-loc | keyword_search | 1 | 0.00 | 0 | D1-1891 |
| E022 | table-loc | keyword_search | 1 | 0.50 | 0 | F5-3241 |
| E023 | table-loc | keyword_search | 1 | 0.00 | 0 | F2-2848 |
| E024 | heading-loc | page_index_search | 1 | 1.00 | 1 | D1-1891 |
| E025 | heading-loc | page_index_search | 1 | 1.00 | 1 | D3-1991 |
| E026 | heading-loc | page_index_search | 1 | 1.00 | 1 | D4-2090 |
| E027 | heading-loc | page_index_search | 1 | 0.50 | 0 | D6-525 |
| E028 | heading-loc | page_index_search | 1 | 0.50 | 0 | D7-2260 |
| E029 | heading-loc | page_index_search | 1 | 0.09 | 0 | The section titled 'D7 AArch64 System Register Descriptions D7.2 General system  |
| E030 | heading-loc | page_index_search | 1 | 0.50 | 0 | D7-2235 |
| E031 | heading-loc | page_index_search | 1 | 1.00 | 1 | D7-2586 |
| E032 | heading-loc | page_index_search | 1 | 1.00 | 1 | D7-2685 |
| E033 | heading-loc | page_index_search | 1 | 0.50 | 0 | E2-2791 |
| E034 | heading-loc | page_index_search | 1 | 1.00 | 1 | F3-2924 |
| E035 | heading-loc | page_index_search | 1 | 0.50 | 0 | F5-3004 |
| E036 | heading-loc | page_index_search | 1 | 0.33 | 0 | F5-3008, F5-3013 |
| E037 | heading-loc | page_index_search | 1 | 1.00 | 1 | F5-3306 |
| E038 | heading-loc | page_index_search | 1 | 0.50 | 0 | F5-3006 |
| E039 | heading-loc | page_index_search | 1 | 0.50 | 0 | F6-3680 |
| E040 | concept | semantic_search | 1 | 1.00 | 1 | D1.20 Interprocessing |
| E041 | concept | semantic_search | 1 | 0.29 | 0 | Access size mismatch |
| E042 | concept | semantic_search | 1 | 0.11 | 0 | Synchronous external abort on an instruction fetch is handled as "Synchronous ex |
| E043 | concept | semantic_search | 1 | 0.10 | 0 | The benefit of using concatenated translation tables is "alignment to the size o |
| E044 | concept | semantic_search | 1 | 0.33 | 0 | Normal memory |
| E045 | concept | semantic_search | 1 | 0.21 | 0 | TLB lockdown: Implementation defined, not required by VMSAv8-64, microarchitectu |
| E046 | concept | semantic_search | 1 | 1.00 | 1 | ARM Limited or its affiliates |
| E047 | concept | semantic_search | 1 | 0.24 | 0 | It depends on the control value:
- 00: Causes trapping unless HCR_EL2.TGE=0 and  |
| E048 | concept | semantic_search | 1 | 0.00 | 0 | FAR_EL1 holds zero (0) if a memory fault comes from a DC ZVA instruction. |
| E049 | concept | semantic_search | 1 | 0.65 | 0 | If HCR_EL2.TID3 equals 1 for non-secure read accesses from EL1, the accesses are |
| E050 | concept | semantic_search | 1 | 0.57 | 0 | The statement applies when EL3 is implemented and using AArch64. |
| E051 | concept | semantic_search | 1 | 0.28 | 0 | The possible values of the MMU enable bit for EL2&0 stage 1 address translation  |
| E052 | concept | semantic_search | 1 | 0.40 | 0 | HVBAR |
| E053 | concept | semantic_search | 1 | 0.71 | 0 | ARM strongly recommends bits ROMADDR[(PAsize-1):32] be zero. |
| E054 | concept | semantic_search | 1 | 0.62 | 0 | Clock frequency indicates the system counter clock frequency. |
| E055 | concept | semantic_search | 1 | 0.00 | 0 | PAN, UAO, SS, IT bits |
| E056 | concept | semantic_search | 1 | 0.22 | 0 | Big-endian data memory accesses |
| E057 | concept | semantic_search | 1 | 0.88 | 0 | Section F1.13 covers Advanced SIMD data-processing instructions. |
| E058 | concept | semantic_search | 1 | 1.00 | 1 | ARM Limited or its affiliates |
| E059 | concept | semantic_search | 1 | 1.00 | 1 | F5.1 Alphabetical list of T32 and A32 base instruction set instructions |
| E060 | concept | semantic_search | 1 | 0.00 | 0 | EL2 |
| E061 | concept | semantic_search | 1 | 1.00 | 1 | ARM Limited or its affiliates |
| E062 | concept | semantic_search | 1 | 0.29 | 0 | ORR<c>.W |
| E063 | concept | semantic_search | 1 | 1.00 | 1 | F5.1 Alphabetical list of T32 and A32 base instruction set instructions |
| E064 | concept | semantic_search | 1 | 0.00 | 0 | LDMIB |
| E065 | table | keyword_search | 1 | 0.83 | 0 | HIFAR is an alias of the Secure IFAR when EL3 uses AArch32 because EL3 is using  |
| E066 | table | read_chunk | 1 | 1.00 | 1 | Outer Shareable |
| E067 | table | keyword_search | 1 | 0.52 | 0 | A status bit value of 1 for the floating-point overflow exception indicates "exc |
| E068 | table | read_chunk | 0 | 0.00 | 0 | YIELD |
| E069 | table | keyword_search | 1 | 0.00 | 0 | n/a |
| E070 | table | read_chunk | 1 | 0.11 | 0 | The type of memory described by code 01 is "Device". |
| E071 | table | keyword_search | 1 | 0.00 | 0 | n/a |
| E072 | table | read_chunk | 1 | 0.00 | 0 | 0 |
| E073 | table | keyword_search | 1 | 0.00 | 0 | VSTR |
| E074 | table | read_chunk | 1 | 1.00 | 1 | SMC |
| E075 | table | keyword_search | 0 | 0.00 | 0 | COND field bit pattern |
| E076 | table | read_chunk | 1 | 0.00 | 0 | UNDEFINED |
| E077 | table | keyword_search | 1 | 0.40 | 0 | Rn: bits 19-16
Rt: bits 15-12 |
| E078 | table | read_chunk | 1 | 0.22 | 0 | 6-10 |
| E079 | table | keyword_search | 1 | 0.00 | 0 | Bits 11 to 7 |
| E080 | table | read_chunk | 1 | 0.00 | 0 | The bit pattern for bits 15 to 11 in the table is "Rd". |
| E081 | table | keyword_search | 1 | 1.00 | 1 | Rn |
| E082 | figure | image_read | 1 | 0.50 | 0 | EL3 |
| E083 | figure | image_read | 1 | 0.00 | 0 | Bits[58:55] |
| E084 | figure | image_read | 1 | 0.19 | 0 | Bits labeled as RES0 are [31:8], bit [5], and bit [0]. |
| E085 | figure | image_read | 1 | 0.50 | 1 | WXN, bit [19] |
| E086 | figure | image_read | 1 | 1.00 | 1 | TXfull |
| E087 | figure | image_read | 1 | 0.00 | 0 | 3 |
| E088 | figure | image_read | 1 | 0.67 | 1 | 0.6 |
| E089 | figure | image_read | 0 | 0.00 | 0 | immediate value |
| E090 | negative | any | 1 | 0.35 | 1 | The document does not contain information about the die area of the processor in |
| E091 | negative | any | 1 | 0.00 | 0 | ARM |
| E092 | negative | any | 1 | 0.40 | 1 | The document does not provide a specific typical power consumption value in mill |
| E093 | negative | any | 1 | 0.31 | 1 | The document does not contain information about the pinout of the JTAG debug con |
| E094 | negative | any | 1 | 0.24 | 1 | The document does not contain information about which physical GPIO pins are use |
| E095 | negative | any | 1 | 0.00 | 0 | Maximum clock frequency: 50 MHz |
| E096 | negative | any | 1 | 0.40 | 1 | The document does not contain information about the unit price of this processor |
| E097 | negative | any | 1 | 0.38 | 1 | The document does not contain information about the operating temperature range  |
| E098 | negative | any | 1 | 0.29 | 1 | The document does not contain information about the package type (BGA/QFP) of th |
| E099 | negative | any | 1 | 0.50 | 1 | The document does not contain information about the number of transistors in thi |