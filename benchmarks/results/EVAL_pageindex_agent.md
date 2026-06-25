# EVAL — page_index + agentic RAG (어려운 20 케이스, 라이브)

> 실제 Upstage·GPT-4.1 mini, 빌드된 ARM v8-A part1 인덱스. 단계별 trajectory와 의도 충족 기록.


**요약: 19/20 케이스 의도 충족.**  C01:✅ · C02:✅ · C03:✅ · C04:✅ · C05:✅ · C06:✅ · C07:✅ · C08:✅ · C09:✅ · C10:✅ · C11:⚠️ · C12:✅ · C13:✅ · C14:✅ · C15:✅ · C16:✅ · C17:✅ · C18:✅ · C19:✅ · C20:✅

## 분석 (반복 개선 후)
- **통과율 16/20 → 17 → 19/20**, 평균 5.4 loops. 세부 trajectory 기록으로 약점을 단계별로 추적해 수정.
- **개선 1 — figure 발견성**: 무번호 figure를 토픽 검색에 노출(합성 캡션=헤딩+페이지 토픽 스니펫) + 같은 페이지 figure 자동 surface → 에이전트가 image_read 트리거.
- **개선 2 — 인용 정확성**: 시스템 프롬프트가 chunk id가 아닌 인쇄 페이지 라벨만 인용하도록.
- **토큰 효율**: page_index_search가 엔트리 ≤6·본문 ≤500자·이미지 ≤4로 캡(페이지 전체 덤프 방지). 캡션에서 '읽는 법' 안내 제거(tool 응답이 안내). 프롬프트에 '최소 호출·실패 검색 반복 금지'.
- **일반화**: 페이지 라벨 추출이 다중 포맷(E2-2804 / 12-3 / Page 45 / 순수숫자) 지원(`tests/test_parser.py`에 회귀). ARM 전용 표현 제거.
- 남은 C11: 답은 정확하나 텍스트로 충분해 image_read 생략(비결정적). 항상 강제하면 토큰 효율과 충돌 → 현 균형 유지.

## C01 [스핀락설계] — ✅ PASS  (loops=8)

**Q**: I'm implementing a spinlock with LDREX/STREX. Read the exclusive-access state diagram and tell me the two monitor states and which instruction enters the exclusive state.


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `page_index_search` {'heading': 'exclusive-access state diagram'} hits=0
- loop2: `keyword_search` {'keywords': ['exclusive-access', 'state', 'diagram']} 
- loop3: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:333', 'ARMv8-Reference-Manual_part1:328']} 
- loop4: `keyword_search` {'keywords': ['local monitor state machine diagram']} 
- loop5: `page_index_search` {'page': 'E2-2804'} hits=22 images=['ARMv8-Reference-Manual_part1_p20_e428.png']
- loop6: `image_read` {'image_path': '/content/local_fast/lsdrag_clean/data/images/ARMv8-Reference-Manual_part1_p20_e428.png'} 
- loop7: `page_index_search` {'heading': 'local monitor state machine diagram'} hits=0
- loop8: `keyword_search` {'keywords': ['local monitor state machine']} 

**의도 충족 체크:**

- ✅ expected tools — used=['image_read', 'keyword_search', 'page_index_search', 'read_chunk'] miss=[]
- ✅ answer keyword — matched=['Open Access', 'Exclusive Access', 'Load-Exclusive']

**답변(발췌)**: Based on the ARMv8 reference manual excerpts and context about exclusive access state machines used with LDREX/STREX instructions:

- The two monitor states relevant to exclusive access are:
  1. Exclusive Access state
  2. Open Access state

- The instruction that enters the exclusive state is the 


## C02 [모니터클리어] — ✅ PASS  (loops=3)

**Q**: For my context-switch code I must know when the exclusive monitor is cleared. What clears it (e.g. CLREX) and what state does it go to?


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `keyword_search` {'keywords': ['exclusive monitor cleared', 'CLREX', 'exclusive monitor state']} 
- loop2: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:347', 'ARMv8-Reference-Manual_part1:259', 'ARMv8-Reference-Manual_part1:220', 'ARMv8-Reference-Manual_part1:222', 'ARMv8-Reference-Manual_part1:224']} 

**의도 충족 체크:**

- ✅ answer keyword — matched=['CLREX', 'clear', 'monitor']

**답변(발췌)**: The exclusive monitor is cleared by the CLREX instruction. According to the ARMv8 reference manual:

- CLREX is the "Clear-Exclusive" instruction.
- It clears the exclusive monitor state.
- The exclusive monitor state goes to "Open Exclusive Access" cleared state (i.e., the exclusive monitor is no l


## C03 [메모리컨트롤러] — ✅ PASS  (loops=8)

**Q**: I'm designing a memory controller. From the memory attributes table, what does the Gathering (G/nG) attribute require me to do?


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `keyword_search` {'keywords': ['Gathering']} 
- loop2: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:79', 'ARMv8-Reference-Manual_part1:33']} 
- loop3: `page_index_search` {'page': 'E2-2788', 'heading': 'Gathering'} hits=13
- loop4: `page_index_search` {'page': 'E2-2788'} hits=24
- loop5: `page_index_search` {'page': 'E2-2788', 'heading': 'Gathering attribute'} hits=0
- loop6: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:38', 'ARMv8-Reference-Manual_part1:39']} 
- loop7: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:40']} 
- loop8: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:41']} 

**의도 충족 체크:**

- ✅ answer keyword — matched=['gather', 'merge', 'combine', 'access']

**답변(발췌)**: The Gathering (G/nG) attribute in the memory attributes table indicates whether memory accesses can be merged (gathered) into a single transaction. Specifically:

- G (Gathering) means the location has the Gathering attribute, allowing multiple memory accesses of the same type (read or write) to the


## C04 [인터커넥트] — ✅ PASS  (loops=3)

**Q**: For my interconnect ordering model, explain the Reordering (R/nR) attribute as given in the attributes table.


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `keyword_search` {'keywords': ['Reordering', 'R/nR', 'attributes table']} 
- loop2: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:102', 'ARMv8-Reference-Manual_part1:103']} 

**의도 충족 체크:**

- ✅ answer keyword — matched=['reorder', 'order']

**답변(발췌)**: The Reordering (R/nR) attribute in the interconnect ordering model indicates whether memory accesses to a location can be reordered or not:

- R (Reordering): The location has the Reordering attribute, meaning memory accesses may be reordered.
- nR (non-Reordering): The location does not have the Re


## C05 [쓰기버퍼] — ✅ PASS  (loops=7)

**Q**: My write buffer design needs the Early Write Acknowledgement hint semantics. What does (No) Early Write Acknowledgement mean and on which page is it? Give the page label.


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `keyword_search` {'keywords': ['Early Write Acknowledgement']} 
- loop2: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:37', 'ARMv8-Reference-Manual_part1:119']} 
- loop3: `page_index_search` {'heading': 'Early Write Acknowledgement'} hits=6
- loop4: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:128', 'ARMv8-Reference-Manual_part1:129']} 
- loop5: `page_index_search` {'page': 'E2-2790'} hits=24
- loop6: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:128', 'ARMv8-Reference-Manual_part1:129']} 

**의도 충족 체크:**

- ✅ expected tools — used=['keyword_search', 'page_index_search', 'read_chunk'] miss=[]
- ✅ answer keyword — matched=['Early Write', 'acknowledge']
- ✅ cites page label — labels=['E2-2790']

**답변(발췌)**: The (No) Early Write Acknowledgement attribute refers to a hint in the memory type attributes for device memory. 

- Early Write Acknowledgement (E) means the location has the Early Write Acknowledgement attribute.
- No Early Write Acknowledgement (nE) means the location has the No Early Write Ackno


## C06 [로컬vs글로벌] — ✅ PASS  (loops=3)

**Q**: I need to implement both a local and a global exclusive monitor in my cache. Compare their behaviour for exclusive access.


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `keyword_search` {'keywords': ['local exclusive monitor', 'global exclusive monitor', 'exclusive access'], 'top_k': 10} 
- loop1: `semantic_search` {'query': 'compare local and global exclusive monitor behavior for exclusive access', 'top_k': 5} 
- loop2: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:231', 'ARMv8-Reference-Manual_part1:239', 'ARMv8-Reference-Manual_part1:251', 'ARMv8-Reference-Manual_part1:316', 'ARMv8-Reference-Manual_part1:325', 'ARMv8-Reference-Manual_part1:357', 'ARMv8-Reference-Manual_part1:359', 'ARMv8-Reference-Manual_part1:417', 'ARMv8-Reference-Manual_part1:227', 'ARMv8-Reference-Manual_part1:236']} 
- loop2: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:241', 'ARMv8-Reference-Manual_part1:315', 'ARMv8-Reference-Manual_part1:301', 'ARMv8-Reference-Manual_part1:355', 'ARMv8-Reference-Manual_part1:245']} 

**의도 충족 체크:**

- ✅ answer keyword — matched=['local monitor', 'global monitor']

**답변(발췌)**: The ARMv8 architecture reference manual describes the behavior of local and global exclusive monitors used for exclusive access in cache-coherent systems. Here is a comparison of their behavior for exclusive access:

Local Exclusive Monitor:
- When a Load-Exclusive instruction is executed by a proce


## C07 [저전력] — ✅ PASS  (loops=8)

**Q**: For a low-power spin-wait loop I want to use WFE/SEV. Which printed page covers Wait For Event / Send Event, and what is the event mechanism?


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `keyword_search` {'keywords': ['Wait For Event', 'Send Event']} 
- loop2: `page_index_search` {'page': 'G1-4403'} hits=0
- loop3: `keyword_search` {'keywords': ['WFE', 'SEV']} 
- loop4: `page_index_search` {'page': 'E2-2804'} hits=22 images=['ARMv8-Reference-Manual_part1_p20_e428.png']
- loop5: `page_index_search` {'page': 'E2-2803'} hits=20
- loop6: `page_index_search` {'heading': 'Use of WFE and SEV instructions by spin-locks'} hits=9 images=['ARMv8-Reference-Manual_part1_p20_e428.png']
- loop7: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:426']} 
- loop8: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:427']} 

**의도 충족 체크:**

- ✅ answer keyword — matched=['Event', 'WFE', 'SEV', 'WFE']
- ✅ cites page label — labels=['E2-2804', 'E2-2804']

**답변(발췌)**: The ARMv8 manual covers the Wait For Event (WFE) and Send Event (SEV) instructions in the section titled "Use of WFE and SEV instructions by spin-locks" on printed page E2-2804. This section explains that WFE and SEV can be used to reduce power consumption and bus contention in low-power spin-wait l


## C08 [검증엔지니어] — ✅ PASS  (loops=3)

**Q**: As a verification engineer I need the exclusive-access state machine to build a checker. Read the figure on page E2-2801 and enumerate the transitions (which ops on which edges).


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `page_index_search` {'page': 'E2-2801'} hits=20 images=['ARMv8-Reference-Manual_part1_p17_e347.png']
- loop2: `image_read` {'image_path': '/content/local_fast/lsdrag_clean/data/images/ARMv8-Reference-Manual_part1_p17_e347.png'} 

**의도 충족 체크:**

- ✅ expected tools — used=['image_read', 'page_index_search'] miss=[]
- ✅ answer keyword — matched=['LoadExcl', 'StoreExcl', 'Exclusive', 'Open']

**답변(발췌)**: The exclusive-access state machine on page E2-2801 has two states: Open Access and Exclusive Access. The transitions with their triggering operations are:

- From Open Access to Exclusive Access:
  - LoadExcl(x, n)
  - StoreExcl(Marked_address, !n)‡
  - Store(Marked_address, !n)
  - StoreExcl(Marked


## C09 [설계리뷰표지] — ✅ PASS  (loops=5)

**Q**: For a design review I must cite sources. Which page label describes Store-Exclusive and the global monitor?


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `keyword_search` {'keywords': ['Store-Exclusive', 'global monitor'], 'top_k': 5} 
- loop2: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:285']} 
- loop3: `page_index_search` {'page': 'E2-2804'} hits=22 images=['ARMv8-Reference-Manual_part1_p20_e428.png']
- loop4: `page_index_search` {'heading': 'Store-Exclusive'} hits=31

**의도 충족 체크:**

- ✅ expected tools — used=['keyword_search', 'page_index_search', 'read_chunk'] miss=[]
- ✅ answer keyword — matched=['E2-28']
- ✅ cites page label — labels=['E2-2800', 'E2-2804']

**답변(발췌)**: The description of Store-Exclusive and the global monitor can be found on page labeled E2-2800 and also on page E2-2804 in the ARMv8 Reference Manual.


## C10 [그림인벤토리] — ✅ PASS  (loops=6)

**Q**: I'm cataloguing the diagrams I need to study. List the figures in this document and the page each is on.


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `keyword_search` {'keywords': ['figure']} 
- loop2: `page_index_search` {'page': 'E2-2797'} hits=20 images=['ARMv8-Reference-Manual_part1_p13_e259.png']
- loop3: `keyword_search` {'keywords': ['figure'], 'top_k': 20} 
- loop4: `page_index_search` {'page': 'E2-2801'} hits=20 images=['ARMv8-Reference-Manual_part1_p17_e347.png']
- loop4: `page_index_search` {'page': 'E2-2797'} hits=20 images=['ARMv8-Reference-Manual_part1_p13_e259.png']
- loop5: `keyword_search` {'keywords': ['figure'], 'top_k': 10} 

**의도 충족 체크:**

- ✅ expected tools — used=['keyword_search', 'page_index_search'] miss=[]
- ✅ answer keyword — matched=['E2-2797', 'E2-2801', 'page']

**답변(발췌)**: The document contains at least two figures with the following details:

- Figure E2-4: Local monitor state machine diagram, located on page E2-2797.
- Figure E2-5: Global monitor state machine diagram for PE(n) in a multiprocessor system, located on page E2-2801.

These are the figures identified by


## C11 [이미지+표통합] — ⚠️ CHECK  (loops=8)

**Q**: For a coherency feature: using BOTH the exclusive-access diagram (read the figure) and the memory attributes table, relate the Exclusive Access state to the Gathering attribute.


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `page_index_search` {'heading': 'Exclusive-access'} hits=0
- loop1: `keyword_search` {'keywords': ['memory attributes']} 
- loop2: `keyword_search` {'keywords': ['exclusive access diagram']} 
- loop3: `keyword_search` {'keywords': ['exclusive access']} 
- loop4: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:239', 'ARMv8-Reference-Manual_part1:325']} 
- loop5: `keyword_search` {'keywords': ['Gathering attribute']} 
- loop6: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:79', 'ARMv8-Reference-Manual_part1:80']} 
- loop7: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:83']} 
- loop8: `keyword_search` {'keywords': ['Exclusive Access state Gathering attribute']} 

**의도 충족 체크:**

- ❌ expected tools — used=['keyword_search', 'page_index_search', 'read_chunk'] miss=['image_read']
- ✅ answer keyword — matched=['Exclusive Access', 'Gather']

**답변(발췌)**: The Exclusive Access state is a condition where the executing processing element (PE) marks a physical memory address for exclusive access, and the local monitor of that PE transitions to the Exclusive Access state. This state is crucial for ensuring that a store instruction succeeds only if the add


## C12 [페이지정밀] — ✅ PASS  (loops=8)

**Q**: I want to bookmark the exact printed page for the exclusive-access state diagram I'll reference in my spec. What is its page label?


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `keyword_search` {'keywords': ['exclusive-access', 'state diagram']} 
- loop2: `keyword_search` {'keywords': ['exclusive-access']} 
- loop3: `keyword_search` {'keywords': ['state diagram']} 
- loop4: `keyword_search` {'keywords': ['state']} 
- loop5: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:333', 'ARMv8-Reference-Manual_part1:328', 'ARMv8-Reference-Manual_part1:355', 'ARMv8-Reference-Manual_part1:245', 'ARMv8-Reference-Manual_part1:251']} 
- loop6: `page_index_search` {'heading': 'state machine for PE(n)'} hits=0
- loop7: `keyword_search` {'keywords': ['Figure E2-5']} 
- loop8: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:342', 'ARMv8-Reference-Manual_part1:350', 'ARMv8-Reference-Manual_part1:353']} 

**의도 충족 체크:**

- ✅ answer keyword — matched=['E2-2801']

**답변(발췌)**: The exclusive-access state diagram you want to reference is Figure E2-5, which is located on printed page labeled E2-2801.


## C13 [전이명령] — ✅ PASS  (loops=6)

**Q**: In the state-machine figure, which single instruction causes the transition from Open Access to Exclusive Access?


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `page_index_search` {'heading': 'state-machine'} hits=0
- loop2: `keyword_search` {'keywords': ['state machine']} 
- loop3: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:254', 'ARMv8-Reference-Manual_part1:262']} 
- loop4: `page_index_search` {'page': 'E2-2797'} hits=20 images=['ARMv8-Reference-Manual_part1_p13_e259.png']
- loop5: `image_read` {'image_path': '/content/local_fast/lsdrag_clean/data/images/ARMv8-Reference-Manual_part1_p13_e259.png'} 

**의도 충족 체크:**

- ✅ expected tools — used=['image_read', 'keyword_search', 'page_index_search', 'read_chunk'] miss=[]
- ✅ answer keyword — matched=['LoadExcl', 'Load-Exclusive']

**답변(발췌)**: The single instruction that causes the transition from Open Access to Exclusive Access in the state-machine figure is the Load-Exclusive instruction (LoadExcl).


## C14 [STREX실패] — ✅ PASS  (loops=8)

**Q**: When does a Store-Exclusive (STREX) fail / not update memory, per the monitor behaviour described here?


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `keyword_search` {'keywords': ['STREX fail', 'Store-Exclusive fail', 'STREX update memory', 'monitor behaviour']} 
- loop2: `keyword_search` {'keywords': ['STREX', 'fail', 'monitor']} 
- loop3: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:285', 'ARMv8-Reference-Manual_part1:290']} 
- loop4: `keyword_search` {'keywords': ['Store-Exclusive fail', 'STREX fail', 'monitor fail', 'exclusive access fail']} 
- loop5: `keyword_search` {'keywords': ['exclusive access instructions', 'Store-Exclusive', 'STREX', 'fail']} 
- loop6: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:231', 'ARMv8-Reference-Manual_part1:237', 'ARMv8-Reference-Manual_part1:282']} 
- loop7: `page_index_search` {'page': 'E2-2796', 'heading': 'Exclusive access instructions and Non-shareable memory locations'} hits=22
- loop8: `page_index_search` {'page': 'E2-2796'} hits=23

**의도 충족 체크:**

- ✅ answer keyword — matched=['fail', 'monitor']

**답변(발췌)**: A Store-Exclusive (STREX) fails and does not update memory if the local monitor for the executing PE is not in the Exclusive Access state for the address being targeted. The local monitor marks the address when a Load-Exclusive is executed, transitioning to the Exclusive Access state. If any other o


## C15 [디바이스메모리] — ✅ PASS  (loops=3)

**Q**: For an MMIO/device region, what does the Device memory attribute section say I must consider? Cite a page label if possible.


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `keyword_search` {'keywords': ['Device memory attribute']} 
- loop2: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:51', 'ARMv8-Reference-Manual_part1:57', 'ARMv8-Reference-Manual_part1:58', 'ARMv8-Reference-Manual_part1:59', 'ARMv8-Reference-Manual_part1:61']} 

**의도 충족 체크:**

- ✅ answer keyword — matched=['Device', 'memory']

**답변(발췌)**: For an MMIO/device region with a Device memory attribute, the manual states several considerations:

- Speculative data accesses are not permitted to any memory location with any Device memory attribute. Each memory access must be generated by a simple sequential execution of the program.
- A write 


## C16 [표참조추적] — ✅ PASS  (loops=4)

**Q**: The Gathering table row points me elsewhere — which page does it tell me to see for more on Gathering?


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `keyword_search` {'keywords': ['Gathering table row']} 
- loop2: `keyword_search` {'keywords': ['Gathering']} 
- loop3: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:33']} 

**의도 충족 체크:**

- ✅ answer keyword — matched=['E2-2788', 'page', '2788']

**답변(발췌)**: The Gathering table row tells you to see page E2-2788 for more on Gathering.


## C17 [범위밖레지스터] — ✅ PASS  (loops=3)

**Q**: I need the SCTLR_EL1 reset value for my boot code. Does THIS document give it? Answer only from this document.


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `keyword_search` {'keywords': ['SCTLR_EL1 reset value']} 
- loop2: `keyword_search` {'keywords': ['SCTLR_EL1']} 

**의도 충족 체크:**

- ✅ graceful (no fabrication) — uncertainty_signal=True

**답변(발췌)**: This document does not contain the SCTLR_EL1 reset value.


## C18 [없는그림] — ✅ PASS  (loops=2)

**Q**: For my coherency block I'm looking for Figure D9-99 on snoop filters — which page is it on?


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `page_index_search` {'figure_no': 'D9-99'} hits=0

**의도 충족 체크:**

- ✅ graceful (no fabrication) — uncertainty_signal=True

**답변(발췌)**: There is no figure labeled D9-99 in the document. If you want, I can help you search for figures related to snoop filters to find the relevant one.


## C19 [신입온보딩] — ✅ PASS  (loops=8)

**Q**: Onboarding a junior engineer: give a grounded overview of the AArch32 synchronization & semaphores material in this doc, citing at least one page label.


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `keyword_search` {'keywords': ['AArch32 synchronization', 'semaphores']} 
- loop2: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:215', 'ARMv8-Reference-Manual_part1:216']} 
- loop3: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:235']} 
- loop4: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:258']} 
- loop5: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:278']} 
- loop6: `page_index_search` {'page': 'E2-2804', 'heading': 'Synchronization and semaphores'} hits=1 images=['ARMv8-Reference-Manual_part1_p20_e428.png']
- loop7: `image_read` {'image_path': '/content/local_fast/lsdrag_clean/data/images/ARMv8-Reference-Manual_part1_p20_e428.png'} 
- loop8: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:215', 'ARMv8-Reference-Manual_part1:216', 'ARMv8-Reference-Manual_part1:235', 'ARMv8-Reference-Manual_part1:258', 'ARMv8-Reference-Manual_part1:278']} 

**의도 충족 체크:**

- ✅ answer keyword — matched=['synchron', 'semaphore']
- ✅ cites page label — labels=['E2-2804', 'E2-2804', 'E2-2804']

**답변(발췌)**: For onboarding a junior engineer on AArch32 synchronization and semaphores, the ARMv8 Reference Manual provides a dedicated section titled "E2.9 Synchronization and semaphores" within the chapter "E2 The AArch32 Application Level Memory Model." This section is found on printed page label E2-2804.

T


## C20 [마킹규칙] — ✅ PASS  (loops=4)

**Q**: For my exclusive monitor RTL, how does a LoadExcl mark the address (what granularity / 'marked address') per this document?


**단계별 trajectory (의도대로 도구를 골랐나):**

- loop1: `keyword_search` {'keywords': ['LoadExcl mark address', 'LoadExcl granularity', 'LoadExcl marked address']} 
- loop2: `keyword_search` {'keywords': ['LoadExcl', 'exclusive monitor', 'marked address']} 
- loop3: `read_chunk` {'chunk_ids': ['ARMv8-Reference-Manual_part1:261']} 

**의도 충족 체크:**

- ✅ answer keyword — matched=['mark', 'address', 'LoadExcl', 'exclusive']

**답변(발췌)**: For your exclusive monitor RTL, a LoadExcl operation marks the address by updating the marked address to the most significant bits of the address X used for the operation. This means the granularity of the marked address is defined by the most significant bits of the LoadExcl address.
