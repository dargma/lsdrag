# BENCHMARK — 독립 골드셋 99문항 (Reader provider = openai)

> 문서: ARMv8-Reference-Manual (expanded parse, data/ir_full). 검색 Recall(출처 검색율) + Reader 최종 답변 F1/EM(SQuAD식).
> 이 셋은 모델-비의존(gold=원문) → **Reader LLM을 바꿔 같은 셋으로 비교**할 수 있다.


## 집계

| 지표 | 값 |
|------|----|
| 검색 Recall (gold 출처 검색율) | **93.94%** |
| 답변 F1 (토큰, gold max) | **0.474** |
| 답변 EM (완전/포함 일치) | **40.40%** |


## 카테고리별 (컨텍스트: 표/그림/텍스트)

| category | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| concept | 25 | 100% | 0.42 | 24% |
| figure | 8 | 88% | 0.37 | 38% |
| figure-loc | 15 | 100% | 0.81 | 73% |
| heading-loc | 16 | 100% | 0.71 | 44% |
| negative | 10 | 100% | 0.29 | 80% |
| table | 17 | 76% | 0.27 | 24% |
| table-loc | 8 | 88% | 0.31 | 12% |

## 도구별 (의도 도구 커버리지)

| tool | n | Recall | F1 | EM |
|----|:-:|:-:|:-:|:-:|
| any | 10 | 100% | 0.29 | 80% |
| image_read | 8 | 88% | 0.37 | 38% |
| keyword_search | 17 | 76% | 0.26 | 12% |
| page_index_search | 31 | 100% | 0.76 | 58% |
| read_chunk | 8 | 88% | 0.32 | 38% |
| semantic_search | 25 | 100% | 0.42 | 24% |

## 문항별

| ID | cat | tool | Recall | F1 | EM | 답변(발췌) |
|----|----|----|:-:|:-:|:-:|----|
| E001 | figure-loc | page_index_search | 1 | 1.00 | 1 | D1-1899 |
| E002 | figure-loc | page_index_search | 1 | 1.00 | 1 | D4-2061 |
| E003 | figure-loc | page_index_search | 1 | 0.09 | 0 | The figure/diagram in the section about 'D7 AArch64 System Register Descriptions |
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
| E015 | figure-loc | page_index_search | 1 | 1.00 | 1 | F5-3569 |
| E016 | table-loc | keyword_search | 1 | 1.00 | 1 | D5-2150 |
| E017 | table-loc | keyword_search | 1 | 0.50 | 0 | D7-2430 |
| E018 | table-loc | keyword_search | 0 | 0.00 | 0 | No exact match found for that sentence in the document. Please provide a shorter |
| E019 | table-loc | keyword_search | 1 | 0.00 | 0 | D7-2590 |
| E020 | table-loc | keyword_search | 1 | 0.50 | 0 | F5-3060 |
| E021 | table-loc | keyword_search | 1 | 0.00 | 0 | arm_p2881_2970:239 contains the table mentioning '5 4 3', but the exact printed  |
| E022 | table-loc | keyword_search | 1 | 0.50 | 0 | F5-3241 |
| E023 | table-loc | keyword_search | 1 | 0.00 | 0 | F2-2848 |
| E024 | heading-loc | page_index_search | 1 | 1.00 | 1 | D1-1891 |
| E025 | heading-loc | page_index_search | 1 | 1.00 | 1 | D3-1991 |
| E026 | heading-loc | page_index_search | 1 | 1.00 | 1 | D4-2090 |
| E027 | heading-loc | page_index_search | 1 | 0.50 | 0 | D6-525 |
| E028 | heading-loc | page_index_search | 1 | 0.50 | 0 | D7-2260 |
| E029 | heading-loc | page_index_search | 1 | 0.50 | 0 | D7-2198 |
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
| E040 | concept | semantic_search | 1 | 0.60 | 0 | Interprocessing is covered in section D1.20. |
| E041 | concept | semantic_search | 1 | 0.29 | 0 | Access size mismatch |
| E042 | concept | semantic_search | 1 | 0.17 | 0 | Synchronous external abort, not on translation table walk |
| E043 | concept | semantic_search | 1 | 0.00 | 0 | Alignment |
| E044 | concept | semantic_search | 1 | 0.33 | 0 | Normal memory |
| E045 | concept | semantic_search | 1 | 0.21 | 0 | TLB lockdown: Implementation defined, not required by VMSAv8-64, microarchitectu |
| E046 | concept | semantic_search | 1 | 1.00 | 1 | ARM Limited or its affiliates |
| E047 | concept | semantic_search | 1 | 0.00 | 0 | HCR_EL2.TGE |
| E048 | concept | semantic_search | 1 | 0.00 | 0 | FAR_EL1 holds zero (0) if a memory fault comes from a DC ZVA instruction. |
| E049 | concept | semantic_search | 1 | 0.38 | 0 | Trapped to EL2 |
| E050 | concept | semantic_search | 1 | 0.50 | 0 | The statement applies when "EL3 implemented and using AArch64." |
| E051 | concept | semantic_search | 1 | 0.00 | 0 | 0, 1 |
| E052 | concept | semantic_search | 1 | 0.40 | 0 | HVBAR |
| E053 | concept | semantic_search | 1 | 0.67 | 1 | ARM strongly recommends that bits ROMADDR[(PAsize-1):32] are zero in AArch32 sys |
| E054 | concept | semantic_search | 1 | 0.62 | 0 | Clock frequency indicates the system counter clock frequency. |
| E055 | concept | semantic_search | 1 | 0.12 | 0 | APSR fields (via MRS and MSR instructions) |
| E056 | concept | semantic_search | 1 | 0.22 | 0 | Big-endian data memory accesses |
| E057 | concept | semantic_search | 1 | 0.78 | 0 | Advanced SIMD data-processing instructions are covered in section F1.13. |
| E058 | concept | semantic_search | 1 | 1.00 | 1 | ARM Limited or its affiliates |
| E059 | concept | semantic_search | 1 | 1.00 | 1 | F5.1 Alphabetical list of T32 and A32 base instruction set instructions |
| E060 | concept | semantic_search | 1 | 0.00 | 0 | EL2 |
| E061 | concept | semantic_search | 1 | 1.00 | 1 | ARM Limited or its affiliates |
| E062 | concept | semantic_search | 1 | 0.29 | 0 | ORR (register) variant |
| E063 | concept | semantic_search | 1 | 1.00 | 1 | F5.1 Alphabetical list of T32 and A32 base instruction set instructions |
| E064 | concept | semantic_search | 1 | 0.00 | 0 | LDMED |
| E065 | table | keyword_search | 0 | 0.58 | 0 | The document does not contain information explaining why HIFAR is an alias of th |
| E066 | table | read_chunk | 1 | 1.00 | 1 | Outer Shareable |
| E067 | table | keyword_search | 1 | 0.00 | 0 | Set |
| E068 | table | read_chunk | 0 | 0.00 | 0 | YIELD |
| E069 | table | keyword_search | 0 | 0.00 | 0 | The document does not contain explicit information on the Accessibility setting  |
| E070 | table | read_chunk | 1 | 0.09 | 0 | The document does not contain information about the type of memory described by  |
| E071 | table | keyword_search | 1 | 0.00 | 0 | n/a |
| E072 | table | read_chunk | 1 | 0.25 | 1 | op1 for CNTP_CVAL_EL0 is 011. |
| E073 | table | keyword_search | 1 | 0.00 | 0 | VSTR |
| E074 | table | read_chunk | 1 | 1.00 | 1 | SMC |
| E075 | table | keyword_search | 0 | 0.00 | 0 | The document does not provide the specific bit pattern for the 'cond' field in t |
| E076 | table | read_chunk | 1 | 0.00 | 0 | UNDEFINED |
| E077 | table | keyword_search | 1 | 0.40 | 0 | Rn: bits 19-16
Rt: bits 15-12 |
| E078 | table | read_chunk | 1 | 0.22 | 0 | 9-5 |
| E079 | table | keyword_search | 1 | 0.00 | 0 | 7-11 |
| E080 | table | read_chunk | 1 | 0.00 | 0 | The bit pattern for bits 15 to 11 in the table is "Rd". |
| E081 | table | keyword_search | 1 | 1.00 | 1 | Rn |
| E082 | figure | image_read | 1 | 0.60 | 0 | The Secure monitor operates at Exception Level EL3. |
| E083 | figure | image_read | 1 | 0.00 | 0 | Bits[58:55] |
| E084 | figure | image_read | 1 | 0.18 | 0 | Bits labeled as RES0 are bits [31:8], bit [5], and bit [0]. |
| E085 | figure | image_read | 1 | 0.50 | 1 | WXN, bit [19] |
| E086 | figure | image_read | 1 | 1.00 | 1 | TXfull |
| E087 | figure | image_read | 1 | 0.00 | 0 | 3 |
| E088 | figure | image_read | 1 | 0.67 | 1 | 0.6 |
| E089 | figure | image_read | 0 | 0.00 | 0 | imm8 |
| E090 | negative | any | 1 | 0.35 | 1 | The document does not contain information about the die area of the processor in |
| E091 | negative | any | 1 | 0.00 | 0 | ARM |
| E092 | negative | any | 1 | 0.40 | 1 | The document does not provide a specific typical power consumption value in mill |
| E093 | negative | any | 1 | 0.36 | 1 | The document does not contain the pinout of the JTAG debug connector. |
| E094 | negative | any | 1 | 0.24 | 1 | The document does not contain information about which physical GPIO pins are use |
| E095 | negative | any | 1 | 0.00 | 0 | Maximum clock frequency: 50 MHz |
| E096 | negative | any | 1 | 0.40 | 1 | The document does not contain information about the unit price of this processor |
| E097 | negative | any | 1 | 0.38 | 1 | The document does not contain information about the operating temperature range  |
| E098 | negative | any | 1 | 0.32 | 1 | The document does not contain information about the package type (BGA/QFP) in wh |
| E099 | negative | any | 1 | 0.50 | 1 | The document does not contain information on the number of transistors in this i |