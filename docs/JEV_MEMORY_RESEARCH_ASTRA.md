# A-RAG + 로컬 Gemma decision: 기억 제어를 새로운 문제 해결 능력으로 연결하기

작성·원문 확인: **2026-10-14 UTC**. 사용자 요청에 따른 Astra 연구 검토. 이 문서는 조사와 실험 설계이며 새 모델 성능을 측정한 결과가 아니다. 조사 범위는 아래에 직접 링크한 논문 본문, 저자 공식 저장소, 공급자 공식 문서다. 최신 연구를 전부 망라했다고 주장하지 않는다.

**우선순위는 근거 충분성 판단 → 제한된 대안 탐색 → 근거의 선택·재읽기 → 출처를 보존하는 장기 기억 갱신이다.** 빠른 decision으로 절약한 시간을 더 좋은 증거를 찾고 검증하는 데 재투자하는 것이 목표다. 단순히 기존 tool name을 더 빨리 고르는 것만으로 기존에 못 풀던 문제가 풀리지는 않는다.

실험 모델은 기존 **`google/gemma-4-E4B-it` Reader의 직접 head 학습, 필요시 같은 Gemma의 추가 fine-tuning**으로 제한한다. Jev, Laya, Kev, SemIf는 구조·학습법의 참고 자료다. 외부 decision API 호출, 다른 decision 모델로의 교체, 외부 teacher API를 사용하는 실험은 이 설계에 포함하지 않는다.

## 1. 현재 결과가 말하는 것과 아직 말하지 않는 것

로컬 근거는 [G1 집계](../lab/reports/search_linked100/m3docvqa/G1_summary.json), [Agent](../src/lsdrag/agent.py), [EvidenceContext](../src/lsdrag/context.py), 완료된 `three_arm_search_linked100/m3docvqa` 궤적이다.

| 관측 | 정확한 범위 | 연구상 의미 |
|---|---|---|
| G1 EM 7%, F1 13.53% | 같은 M3DocVQA 100문항 | 정답 생성까지의 품질이 낮다. 원인을 한 가지로 단정할 수 없다. |
| Figure candidate/open/delivery recall 모두 54.29% | 시각 근거 매핑이 있는 **70문항** | 후보로 잡힌 정답 그림의 open·delivery 단계에서 집계상 손실이 없다. 100문항 EM과 직접 빼거나 나눠 조건부 정답률을 만들면 안 된다. |
| Document candidate recall 79.48%, read recall 70.17% | 100문항 | 후보 검색 이후 full evidence 접근에도 개선 여지가 있다. |
| 모든 정답 문서 candidate 확보 66%, read 확보 54% | 100문항 | 일부 문서만 얻고 종료하는 multi-document 실패를 따로 조사할 가치가 있다. |
| G1 평균 1.82 tool calls, 2.82 Reader calls | 100문항 | 최대 8 tool loops를 대부분 소진하지 않았다. 기존 실패를 모두 loop cap 탓으로 돌릴 수 없다. |
| B0/C1/G1 성공 궤적 2/0/7 | 총 300궤적, 서로 다른 300문항이 아님 | 성공 증류의 독립 문제 수와 class coverage가 극히 작다. |
| Reader 호출 339/334/282, 합계 955 | 직접 로그 검사 | 모든 호출이 단일 tool 또는 FINAL. 복수 tool 호출 예제는 없다. |

후속 [고정 split](../lab/registry/jev_v1.0/split.json)과 [teacher audit](../lab/reports/jev_v1.0/teacher_audit.json)은 outcome-independent seed `20261014`, train/dev/test **60/20/20**을 사용한다. train 성공은 **5궤적·4개 독립 qid·15턴**이며 keyword 5, semantic 5, FINAL 5, **read 0**이다. dev는 성공 3 qid·11턴, test는 성공 1 qid·5턴이다. 따라서 현재 head는 학습·저장·추론 **동작 진단만 Go**, 일반적인 4-class policy 평가·calibration 주장은 **No-go**다. [실행 계획](JEV_EXPERIMENT_PLAN.md)의 신규 train 최대 100문항 수집과 사전 gate(성공 독립 문항 30개 이상, 클래스마다 10개 이상 독립 성공 문항에서 관측)를 따른다. 이 gate는 통계적 충분성을 보장하지 않는다.

현재 corpus는 **288문서로 축소된 코퍼스이며 공식 전체 open-domain corpus가 아니다**. 이 288문서 안에서는 추가적인 question-specific 검색 범위 제한을 두지 않는다. **준비된 corpus 접근 가능성**, **검색 top-k**, **Reader context**, **8개 이미지 예산**, **loop/time budget**은 서로 다른 제한이다. `read_chunks`, `delivered_hashes`는 현재 질문의 근거 추적 장치이며, 사실을 수정·버전 관리하는 장기 기억 저장소는 아니다.

먼저 실패를 다음처럼 다시 분류해야 한다. 아래는 **점검할 가설이지 이미 측정된 실패 빈도는 아니다**.

1. 정답 문서/그림이 후보에 없음: query·검색 경로·분기 문제.
2. 후보는 있으나 필요한 전체 내용/관계가 없음: read·인접 chunk·coverage 문제.
3. 필요한 근거가 실제 전달됐으나 답이 틀림: 이미지 해석·엔티티 구별·다중 근거 결합·답변 형식 문제.
4. 그럴듯한 근거만 보고 종료: 충분성 오판·다른 해석 미탐색 문제.
5. 오래된 사실을 현재 사실로 사용: 장기 기억 확장용 실패 유형. 현재 정적 M3 결과로 발생률을 주장할 수 없음.

## 2. 최신 연구에서 실제로 바뀌는 관심사

아래 날짜는 arXiv **제출일** 또는 명시된 버전 날짜다. 검색엔진의 발표·크롤링 날짜와 구분했다. 논문 결과는 저자 보고이며 이 저장소에서 재현한 것이 아니다.

| 연구·확인 버전 | 원문에서 확인한 방향 | A-RAG에 가져올 관점과 한계 |
|---|---|---|
| [A-RAG](https://arxiv.org/abs/2602.03442), 2026-02-03 | keyword/semantic/chunk read를 모델에 직접 노출하는 계층적 검색 인터페이스 | 기존 도구를 유지한 채 **제어 정책**을 비교할 기반. 큰 모델의 성과가 작은 Gemma head로 자동 이전되지는 않는다. |
| [AgeMem](https://arxiv.org/html/2601.01885v1), 2026-01-05 v1 | 장기·단기 기억의 저장·검색·정리 행동을 함께 다루고 단계적 RL로 학습 | 도구 선택과 context 관리의 결합이 연구 대상. 현재의 9개 성공 궤적으로 이런 RL을 재현하는 것은 부적절하다. |
| [MemoryArena](https://arxiv.org/html/2602.16313v1), 2026-02-18 v1 | 이전 세션의 행동·피드백을 기억해 다음 세션 행동에 활용하는 평가 | 단일 질문 recall 향상과 장기 기억 능력을 구분한다. M3 단독 결과로 long-term-memory 개선을 주장하지 않는다. |
| [TrustMem](https://arxiv.org/html/2606.25161v1), 2026-06-23 v1 | Write/Revise/Prune 전후의 coverage·preservation·faithfulness와 동일 상태 후보 순위 학습 | memory write의 결과뿐 아니라 **전이**를 검증. 자세한 검토는 [기존 문서](TRUSTMEM_PAPER_REVIEW.md). 이미지 기억으로의 적용은 새 가설이다. |
| [MemChain](https://arxiv.org/html/2607.24097v1), 2026-07-27 v1 | retrieval을 고정하고, 검색 후보를 근거 ID에 연결된 답변용 evidence로 변환하는 policy; SFT 후 TMPO | recall과 reader 입력의 질을 분리하는 대조 설계가 유용하다. 긴 trace 생성 자체는 우리의 속도 목표와 충돌할 수 있어 ID 선택부터 시작한다. |
| [AgenticRag-R1](https://arxiv.org/html/2608.29622v1), 2026-08-30 v1 | stack memory, 세분화한 행동, 행동별 보상과 정보 기반 rollout 선택 | 검색·기억·추론의 상태 전이가 학습 대상. 추가 행동과 rollout 비용을 숨기지 않아야 한다. |
| [Gated-Memory Routing](https://arxiv.org/html/2609.00237v2), 최초 2026-08-31, v2 09-09 | write gate, retrieval gate, halting을 함께 제어 | 유사도만으로 다음 context를 구성하지 않는 방향. multi-agent 성과를 단일 Gemma 성과로 옮겨 주장하지 않는다. |
| [Jev-Mem](https://arxiv.org/html/2609.23986v1), 2026-09-21 v1 | 기억 관계·검색 경로·예산·후보·종료를 typed control로 분리 | 직접적인 선행 연구. 아래 재현상 주의점을 반드시 함께 읽어야 한다. |
| [MemCalib](https://arxiv.org/html/2609.24259v3), 최초 2026-09-21, **v3 10-08** | 기억을 너무 많이/적게 사용하는 오류를 나누고 원자적 사실 제거에 따른 응답 likelihood 변화로 credit 배분 | relevance gate만으로 답변 영향이 적절하다고 볼 수 없다. 우리 실험에서는 전체 RL 대신 controlled evidence removal을 진단에 사용한다. 이름의 calibration은 단순 temperature calibration과 다르다. |
| [TRACE](https://arxiv.org/html/2609.33517v1), 2026-09-27 v1 | 중단 후 돌아온 agent가 기존 기억을 현재 상태에서 다시 사용할 수 있는지 판정 | retrieval relevance와 **현재 유효성**이 별개임을 보여 주는 설계. 단일 문서 QA가 아니라 상태가 변하는 후속 벤치마크에 적합하다. |
| [RIME](https://arxiv.org/html/2609.34438v1), 2026-09-28 v1 | write 시 근거를 검색하고, 압축 기억이 부족하면 원 대화와 주변 문맥을 다시 읽음 | memory는 원본을 대체하지 않는 인덱스·요약이 될 수 있다. 그림 요약만으로 단정하지 않고 원본 픽셀로 돌아가는 가설과 연결된다. |
| [What Should an Agent Remember?](https://arxiv.org/html/2610.00366v1), 2026-09-30 v1 | retention과 retrieval selection을 같은 접근 조건에서 교차 비교 | 버린 정보는 좋은 retriever도 복구할 수 없다. 작은 합성·단일 recall 환경의 결과이므로 일반 agent 성능 수치로 확대하지 않는다. |
| [The Right Memory in the Wrong Context](https://arxiv.org/html/2610.07309v1), **2026-10-05 v1** | relevance·admissibility·prompt exposure·최종 사용을 구분하고 필요한 근거 recall을 맞춰 비교 | entity/time/version 적용 가능성도 별도 검증할 이유. 원문의 namespace/policy 문제를 현재 M3에 존재하는 문제로 단정하지 않는다. |

이 자료들에서 읽어 낸 공통 방향은 세 가지다. 첫째, 검색 뒤에 근거를 정리하고 필요성을 판단하는 **별도 단계**가 생긴다. 둘째, 기억의 추가·수정·삭제를 최종 QA 점수만으로 평가하지 않는다. 셋째, 좋은 검색 결과라도 현재 질문과 시점에 맞게 사용할 수 있는지 평가한다. 이 종합은 위 자료들에 근거한 본 문서의 해석이다.

### Jev-Mem을 그대로 성능 근거로 쓰면 안 되는 이유

[저자 저장소](https://github.com/libingzheren/Jev-Mem)는 2026-09-27 Laya 통합을 알리지만 **논문 수치는 Jev 사용 결과이며 Laya는 해당 논문에서 평가하지 않았다**고 명시한다. 기본 inherited `best-of-n=3`에는 reference-aware selection이 있어 `best-of-n=1`을 권한다. construction timer는 모델 초기화와 저장 시간을 제외하고, LongMemEval runner는 공식 metric과 다른 scorer를 쓴다. 따라서 공개 runner 실행을 곧 논문 재현이라고 부를 수 없다.

원문 Table 1의 temporal 점수는 Jev-Mem 0.637, MAGMA 0.650인데 본문은 최고와 같다고 서술한다. 문서 내부 불일치가 있으므로 세부 수치를 채택할 때 표·코드·출력을 재검증해야 한다. 이 메모에서는 **typed control 구조만 참고**하고 저자 속도·정답률을 로컬 Gemma의 예상 효과로 사용하지 않는다. [논문 §4.2](https://arxiv.org/html/2609.23986v1#S4.SS2)

## 3. Jev·Laya·Kev·SemIf: 공개된 학습 방법의 차이

확인일은 모두 2026-10-14다. 저장소 `main`/`master`는 변하므로 실제 구현에 채택할 때 commit과 코드 hash를 별도로 고정해야 한다. 아래는 채택·실행 결과가 아니라 **열람한 공개 구현 비교**다.

| 계열 | 공개 범위·학습 방식 | 이 프로젝트에서의 해석 |
|---|---|---|
| **Jev / TypeSafe** | [공식 AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)는 RLCD를 calibrated decisions용 post-training으로 설명한다. 확인한 공식 페이지는 정확한 loss·reward·data·가중치·전체 training recipe를 제공하지 않는다. | 공개 설명만으로 RLCD를 재현했다고 주장하지 않는다. “새 architecture/sampler”와 속도 수치는 공급자 주장이다. `jev.org.cn`, `typesafe-ai.com` 같은 별도 소개 사이트를 TypeSafe 공식 문서로 인용하지 않는다. |
| **Laya** | [공식 repo](https://github.com/NandhaKishorM/laya)는 encoder 기반 decision model. [현재 train.py](https://github.com/NandhaKishorM/laya/blob/main/laya/train.py)는 `rlcd`와 `soft-ce`를 제공. 전자는 noisy logits에 대한 Gaussian policy-gradient 항, proper reward, soft CE를 결합한다. 코드 주석은 typed-decisions 실험에서 추가 RL 항의 이득이 없었다고 기록한다. | Laya의 독립 구현은 TypeSafe 내부 RLCD와 동일하다는 증거가 없다. 선택 순서 augmentation, soft targets, 별도 calibration을 참고하되 초기 Gemma 실험은 단순 CE를 기준선으로 둔다. |
| **Kev-4B** | [공식 model card](https://huggingface.co/jaredpalmer/kev-4b/blob/main/README.md): Qwen3.5-4B-Base 위 rank-16/α32 LoRA와 pointer head, 선택지 CE. 선택지 순서 변경, none-of-the-above, 정답 선택지·결정 근거 제거 예제를 포함한다. 별도 dev 1,264 questions로 NLL을 최소화하는 단일 temperature T=2.41을 맞춘다. | backbone adaptation과 작은 readout을 함께 학습하는 공개 사례. API 모양이 같아도 Jev 학습 복제가 아니다. Gemma용 LoRA·pointer의 feasibility를 따로 검증해야 한다. |
| **SemIf** | [원저자 repo](https://github.com/TheoLeeCJ/SemIf-OpenJev)는 frozen model의 후보 token logits를 직접 읽고 prefix 재사용을 비교한다. 2026-09-22 기록에는 별도의 workload temperature calibration이 추가되었다. | **무학습 readout**과 **head 학습**을 구별하는 적절한 대조군. 원문도 generated baseline과 선택이 완전히 같지 않다고 밝혀 시스템 속도 비교와 품질 비교를 분리한다. |

Jev의 `confidence`도 “별도 정답확률 예측기”라고 오해하면 안 된다. 공식 [Confidence 문서](https://docs.typesafe.ai/confidence)는 Choice에서 `(p_max - 1/n)/(1 - 1/n)`라는 분포 요약값을 반환한다고 설명한다. 이 값, raw `p_max`, teacher 행동과의 일치 확률, 최종 답 정답확률은 서로 다르다. 우리 head는 **무슨 사건의 확률인지 먼저 정의**해야 한다.

### 로컬 Gemma에 권하는 학습 경로

1. **Frozen Gemma + 정규화 linear head + CE**: 기존 multimodal 상태의 generation boundary hidden state로 4개 행동을 분류한다. 이것이 가장 작은 검증 단위다.
2. **같은 상태에서 수집한 행동 비교 + 근거 반사실 예제**: 선택한 행동을 따라가 본 결과로 유용성을 비교하고, entity/date/필수 근거가 바뀌었을 때 판단도 바뀌는지 학습한다. train에서만 gold를 보상·감사에 사용한다.
3. **필요할 때만 같은 Gemma에 LoRA**: frozen feature로 구별되지 않는 오류가 확인되고 독립 train 질문·클래스가 충분해진 뒤 검토한다. answer path에도 adapter를 적용한다면 그 자체가 추가 변경이므로 adapter on/off와 final-answer 성능을 따로 평가한다.
4. **dev calibration은 데이터가 충분할 때만**: 성공 9궤적으로 calibration과 threshold 선택까지 안정적으로 할 수 있다고 가정하지 않는다. 보정 전 NLL/Brier와 적은 표본 수를 먼저 공개한다. proper scoring loss도 제한된 데이터·분포 이동에서 보정 성능을 보장하지 않는다.

동일 state에 10개 gate를 붙이는 것은 head 연산만 보면 저렴할 수 있다. 그러나 gate마다 별도 question suffix로 full Gemma prefill을 다시 수행하면 저렴하지 않다. 현재 Reader의 최종 hidden state에 공유 multi-head를 붙일지, candidate를 포함해 별도 forward할지 실제 비용을 계측해야 한다.

## 4. 우선순위별 검증 가능한 다섯 가설

아래 H1–H5는 **제안된 가설**이다. 논문에서 Gemma+M3 조합으로 입증한 결과가 아니며, 현재 원인 감사 전에 실패 유형의 비율을 알 수 없다. 숫자로 적은 실험 크기·예산은 시작 설계값이다.

### H1. 충분성·충돌 gate가 조기 종료를 줄여 다중 근거 문제를 추가로 푼다 — 최우선

**대상 실패:** 관련 문서 하나를 찾았지만 질문의 나머지 관계·시각 속성·비교 대상이 비어 있는 상태에서 FINAL. G1의 all-documents candidate 66%와 all-documents read 54% 차이는 이 조사의 동기이지 원인 증명은 아니다.

**행동:** 기존 다음-tool head와 별개로 `sufficient / missing / conflicting / unknown`을 판정한다. unknown은 missing과 같지 않다. 코드는 남은 시간·미읽은 ID·반복 행동을 함께 보고 `FINAL`, `READ_NEXT`, `SEARCH_MISSING`, `CHECK_CONFLICT`를 실행한다. 구체적인 query는 질문·발견한 entity·근거에서 로컬 Gemma가 생성하거나 정해진 후보 중 고른다. gate만으로 새로운 query가 생기지 않는다.

**추가 능력 가설:** 단일 후보에 성급히 만족하는 대신 부족한 연결고리를 한 번 더 읽어 다중 문서 답을 완성한다. decision 시간 절감이 없더라도 gate 자체의 품질 개선 여부를 비교한다.

**최소 실험:** train의 서로 다른 20문항에서 첫 FINAL 직전 state를 저장하고, 같은 state에서 `지금 답하기`와 `허용된 추가 행동 1회 후 답하기`를 비교한다. gold 근거는 결과 평가·train label 수확에만 사용하고 policy 입력에는 넣지 않는다. 추가 행동은 실제 기존 corpus에서 실행하고 결과·arguments를 보존한다. 이 diagnostic으로 학습하고 dev에서 설정을 고른 뒤, label gate가 충족되면 outcome-independent heldout 20문항을 처음부터 돌린다.

**대조·측정:** 원래 A-RAG, 무조건 1회 더 검색, 짧은 생성형 verifier, compact gate. EM/F1, 완전 evidence 확보율, 기존 실패→성공/기존 성공→실패, 잘못된 충분 판정, 불필요한 추가 행동, 총 시간을 함께 측정한다. gate 운영비는 우선 총 시간 예산의 10% 이하 목표로 두되 실측 달성 여부를 보고한다. 항상 CONTINUE하는 정책이 좋은 gate로 보이지 않게 한다.

### H2. 제한된 대안 분기가 잘못된 entity·검색 해석에서 회복시킨다 — 두 번째

**대상 실패:** 첫 keyword/semantic 경로가 비슷한 이름의 다른 entity나 관련 장소·그림에 고착되고 후속 검색도 같은 후보를 반복한다. 후보에 gold figure가 없는 문항에서 우선 원인을 확인한다.

**행동:** `(tool, concrete arguments)` 후보를 최대 3개 만든다. 예: 원 질문 semantic query, 발견한 정확한 entity 중심 keyword query, 아직 읽지 않은 문서/인접 chunk read. 한 상태의 후보들을 점수화해 최대 **2개 분기, 깊이 1**만 탐색하고 추가 evidence gain·중복·충돌·남은 비용으로 계속할 분기를 고른다. 각각의 retrieval/context 상태는 독립 snapshot으로 유지한다.

**추가 능력 가설:** 한 번의 잘못된 검색 해석을 되돌려 다른 증거 경로를 확보한다. 다수결 답변보다 **다른 근거를 발견하는 것**이 목표다. 모든 분기가 같은 틀린 전제를 공유하면 효과가 없다.

**최소 실험:** train의 20개 search state에서 동일 후보 집합으로 original-first, random/round-robin, relevance top-1, decision top-2를 비교한다. 이후 heldout에서는 candidate 생성부터 모두 실행한다. GPU상 두 VLM 분기를 동시에 실행하지 않아도 논리적 branching은 가능하다. shared GPU의 물리적 병렬화는 자원 조건을 만족할 때 별도로 측정한다.

**대조·측정:** equal-time A-RAG의 추가 loop, 단순 two-query union, head가 선택한 분기. 신규 gold document/figure 발견율, 후보 다양성, chosen-branch regret(사후 평가), branch 유지 비용, 최종 EM/F1. 추가 후보 생성·모든 탈락 분기 비용도 시간에 포함한다. 예산의 탐색 부분을 초과하면 기존 분기의 근거로 종료하며 최고 gold 점수를 가진 branch를 답으로 고르는 oracle selection은 금지한다.

### H3. 근거를 역할별로 선택하고 원본을 재읽으면 이미 검색한 정보로도 더 잘 답한다 — 세 번째

**대상 실패:** 필요한 근거가 전달됐어도 distractor, entity 혼동, 관계 누락, 이미지 캡션의 부정확한 요약 때문에 답하지 못한다. 이 유형은 실제 전달된 ID 기준으로 확인해야 하며 EM과 recall의 집계 차이만으로 사례를 만들면 안 된다.

**행동:** 검색 후보에 `support / bridge / contradiction / distractor / unknown` 역할을 매기고, `KEEP`, `PROMOTE_TO_FULL_READ`, `DEFER`, `REHYDRATE_SOURCE`를 선택한다. 초기에는 텍스트를 새로 요약하지 않고 원문 span·chunk ID·figure ID를 선택해 생성 오류를 줄인다. 시각 속성은 캡션만으로 검증했다고 처리하지 않는다.

**추가 능력 가설:** 질문에 필요한 연결 근거를 유지하면서 방해 내용을 줄여, 같은 candidate set으로도 정확한 결합이 가능해진다. 이미지 후보를 찾는 능력과 전달된 이미지를 이해하는 능력을 분리해서 측정할 수 있다.

**최소 실험:** 20개의 실제 최종 state에서 candidate와 픽셀을 고정하고 `모두 입력`, `retrieval score 순 입력`, `compact role selection`을 비교한다. 근거를 뺀 조건과 entity만 바꾼 distractor 조건을 추가해 모델이 무엇에 의존하는지 관찰한다. 원본 re-read를 허용하는 실험은 candidate-fixed 실험과 별도로 보고한다.

**대조·측정:** fixed-candidate EM/F1, 필수 bridge 보존율, false exclusion, citation 정확성, 이미지 포함/제외 시 답의 변화, 총 prefill tokens·시간. context에서 제거한 이미지와 질문 내 총 unique image cap을 구별한다. 이미지 eviction으로 8개보다 더 많은 새로운 이미지를 누적 읽게 하면 자원 조건 변경이므로 별도 arm으로 둔다. 이미지 재읽기가 같은 비용 없이 이루어진다고 가정하지 않는다.

### H4. 출처와 버전을 보존하는 write/revise/prune이 후속 질문의 오류 전파를 줄인다 — 장기 기억 확장

**대상 실패:** 이전 답변·요약의 추측이 사실로 저장되거나, 수정된 사실의 옛 버전이 이후 답을 지배하는 문제. 현재 M3는 정적 corpus QA이므로 이 능력을 평가할 별도 stream이 필요하다.

**행동:** immutable source를 유지하고 메모리 레코드에 `source_id`, `chunk/span`, `figure_sha256`, `entity`, `observed_at`, `valid_from/to`, `supersedes`, `status`를 둔다. 시간 값이 원문에 없으면 모름으로 남긴다. Gemma head는 proposed update에 대해 `WRITE / REVISE / KEEP_BOTH / NOOP / PRUNE_DERIVED`를 선택한다. 내용이 새로 필요한 WRITE/REVISE는 로컬 Gemma가 생성하고 출처를 연결한다. conflict가 해결되지 않았으면 최신이라는 이유만으로 덮어쓰지 않는다.

**추가 능력 가설:** 한 번 읽은 사실을 재사용하면서 정정·역사 시점 질문에 각각 맞는 버전을 찾고, 잘못된 메모리 하나가 여러 후속 답을 망치는 현상을 줄인다. source 원본은 prune하지 않으므로 필요할 때 재구성할 수 있다.

**최소 실험:** train과 분리한 entity로 20개의 합성 stream을 만들고 stream마다 6개 관측과 4개 후속 질문을 고정한다. 각 stream에 날짜 있는 정정, 다른 entity의 동명 사실, 단순 반복, 미해결 충돌 중 일부를 포함한다. “현재”, “정정 전”, “알 수 없음” 질문을 섞는다. generator·seed·정답·version graph를 공개하며 이는 M3 성능을 대신하지 않는 unit benchmark로 명명한다. 이후 LongMemEval의 update/temporal/abstention 또는 실제 연속 문서로 확장한다.

**대조·측정:** append-only hybrid retrieval, last-write-wins, 생성형 consolidate, head-controlled update. valid-fact retention, 잘못된 overwrite, 근거 없는 write, stale-fact use, 후속 QA, 총 build+query 시간. write/revise verifier 비용과 미래 query 수에 따른 amortization을 따로 보고한다. admission gate가 덜 저장해서 빨라진 효과와 retrieval 향상을 분리한다.

### H5. 위험에 맞춘 추가 검증·보류가 과신을 줄이고 절약한 시간을 어려운 문제에 재배분한다 — H1 이후

**대상 실패:** evidence 부족·충돌인데 FINAL을 강행하거나, 쉬운 상태에서 반복 검색해 어려운 질문에 쓸 시간을 낭비하는 문제. 현재 성공 라벨만으로 정답확률 predictor를 학습할 수 없다.

**행동:** `ANSWER / VERIFY_ONCE / CONTINUE / ABSTAIN`을 선택한다. 행동 distribution, evidence coverage, conflict flags, 비용을 입력으로 사용하되 raw max-softmax를 곧 correctness probability라고 부르지 않는다. 부족한 근거와 candidate가 모두 부적절한 상태를 train 예제로 포함한다.

**추가 능력 가설:** 잘못된 자신감의 답을 줄이면서, 아직 회복 가능한 질문에는 추가 탐색을 할당한다. 보류는 새로운 정답을 만든 것이 아니므로 answered accuracy만 좋아졌다는 이유로 능력 향상이라고 보고하지 않는다.

**최소 실험:** train/dev에서 결정 근거 제거·모순 삽입 minimal pair를 만들고 알려진 answerability를 label로 쓴다. test에서는 동일 질문 원본/변형을 함께 묶어 평가하며 threshold는 고정한다. 초기 20문항은 동작 확인에만 쓰고 보정된 위험 보장을 주장하지 않는다.

**대조·측정:** 항상 답변, random abstention, 단순 coverage rule, calibrated head(가능할 때). 전체 문항 EM/F1, risk–coverage 곡선, answerable false-abstention, unanswerable false-answer, 회복한 문항 수, 시간. 무조건 보류하는 정책의 높은 answered accuracy가 결과를 부풀리지 않게 coverage와 전체 분모를 고정한다.

## 5. 공동 실험 규약: 속도가 능력으로 전환됐는지 검증하기

### 5.1 같은 시간과 같은 접근권을 비교한다

**사전 dev 측정으로 동일 backend baseline의 문항당 latency 분포를 얻고**, 고정 시간 예산 `B`를 정한다. 예를 들어 dev median, dev p75, 그 두 배의 세 점으로 quality–time 곡선을 측정할 수 있다. 각 arm은 동일한 `B`를 사용하며 test 문항의 baseline 실행 시간을 보고 개별 예산을 정하지 않는다. 기존 G1 평균 8.87초는 과거 vLLM 측정치이므로 새로운 HF 예산으로 그대로 가져오지 않는다.

시간은 아래 항목을 모두 포함한다.

```text
T_total = state/template/image preparation
        + decision prefill/readout
        + candidate/query/argument generation
        + retrieval and all explored branches
        + read/vision encoding/context reconstruction
        + verification/fallback
        + final answer generation
```

최종 답변에 쓸 reserve는 dev에서 정하고 모든 arm에 적용한다. deadline을 넘긴 결과도 버리지 말고 overrun·timeout으로 기록한다. operation 사이에서만 시간을 검사하는 구현은 strict preemption이라고 부르지 않는다. wall time 외에 GPU work 또는 총 model forward·tokens도 함께 기록해 병렬 fanout의 추가 자원을 드러낸다. cold/warm·cache 조건, 실행 순서, batch/concurrency를 고정하고 외부 GPU activity를 기록한다.

두 비교를 분리한다. **행동 동일·same-state 비교**는 decision 비용과 imitation 품질을 측정한다. **closed-loop equal-budget 비교**는 정책이 상태를 바꾼 뒤의 새 정보 획득·최종 답을 평가한다. 같은 loop 수에 더 비싼 분기를 숨기거나 HF compact를 vLLM baseline과 비교해 head의 속도 이득으로 해석하지 않는다.

### 5.2 counterfactual은 동일 이전 상태에서 시작한다

1. `state_id`를 질문·history·tool schema·retrieval snapshot·image hashes·memory version으로 정의한다.
2. 같은 state에서 합법적인 다른 행동을 실제 실행한다. 후보와 arguments 생성에는 gold answer/figure/document ID를 제공하지 않는다.
3. 이후 budget·Reader·seed·index를 맞춰 답을 생성한다. train에서만 gold 기반 결과를 teacher preference에 이용한다.
4. branch가 이미 달라진 뒤 같은 loop 번호라는 이유로 pair를 만들지 않는다.
5. fixed-state evidence removal과 full rollout action intervention을 구분한다. evidence 제거 후 응답이 바뀌었다는 사실만으로 원래 행동의 인과적 유용성을 확정하지 않는다.

기존 deterministic greedy 로그에는 선택되지 않은 행동의 support가 없다. 따라서 오프라인 tool accuracy나 importance weighting만으로 새 policy의 closed-loop value를 추정할 수 없다. 로그 replay는 먼저 어디를 실제 재실행해야 하는지 찾는 진단이다.

### 5.3 split·label·calibration의 제한

qid group split을 outcome을 보기 전에 seed로 동결한다. 세 arm의 같은 qid와 minimal-pair 변형은 함께 묶고, 문서/동일 entity 중복률을 공개한다. 성공 qid를 전부 train으로 먼저 옮기면 남은 test가 기존 baseline 실패에 조건화된다. 이것은 대표 test가 될 수 없다. 새 outcome-unseen test를 추가할 수 있다면 기존 100문항을 개발 자원으로 쓰는 별도 설계가 가능하다.

성공-only teacher는 쉬운 질문과 성공한 검색 경로에 치우친다. EM 성공도 모든 중간 행동의 최적성을 보장하지 않는다. 처음에는 class별 independent question 수를 보고하고 fixed 4-class head의 feasibility를 평가한다. 실패 state의 충분성·충돌·반사실 label을 추가할 때는 “성공 궤적 증류”와 다른 supervision source임을 명시한다.

20문항 test는 사례·실행 검증에 적합하지만 작은 정답률 차이, p95, calibration 안정성을 증명하기에는 부족하다. `recovered failures`, `lost successes`, question 단위 paired 차이와 불확실성을 공개하고 개선폭을 보고 난 뒤 test를 재선별하지 않는다.

### 5.4 구현에서 지켜야 할 경계

현재 tool 호출은 자유 arguments를 포함한다. `keyword_search` 결정만으로 keywords가 생기지 않고, `read_chunk` 결정만으로 올바른 IDs가 선택되지 않는다. **고정 행동 head**, **동적 후보 pointer**, **argument generator**를 별도 구성 요소로 명명한다. candidate-only 정책은 행동 공간을 제한하므로 candidate generator의 coverage도 측정한다.

Gemma의 전체 multimodal forward를 유지해야 한다. local implementation에는 PLE와 vision mask가 있어 text-only decoder를 대용으로 호출하면 같은 상태가 아니다. 최종 normalized last-token feature만 저장하고 42개 층 전체 hidden tensor를 보존하지 않는다. KV 재사용은 실제 native template의 prefix 일치·image 처리·cache 변형을 검증한 뒤 사용한다. 선택지 instruction을 새로 붙인 zero-shot prompt는 evidence가 같아도 byte-identical prompt는 아니다.

고정 분류 head는 전체 검색·읽기·수정 내용을 만들어 주지 않는다. 새 정보가 corpus나 관측에 없으면 더 빠른 head도 만들어 낼 수 없다. 정보가 있는데 Reader가 시각 관계를 해석하지 못한다면 policy만 바꿔도 회복되지 않을 수 있다. 이 한계는 실제 검색·전달된 근거의 감사와 고정 candidate 실험으로 조사한다. **정답 근거를 Reader에 직접 제공하는 oracle-evidence 조건은 사용자 지시에 따라 이번 실험에서 제외한다.**

## 6. 진행 기준과 예상 기여

| 단계 | 수행할 일 | 다음 단계로 갈 증거 |
|---|---|---|
| A: 원인 감사 | question별 candidate→read→delivery→answer와 첫 FINAL state 재구성 | 최소 두 실패 유형에서 실제 ID·이미지·답변을 연결한 사례 확보 |
| B: decision feasibility | 15턴의 학습·저장·추론 동작 진단, 무학습 readout 대조, native arguments 경로 | 입력·parse·memory·비용 계측 정상. 일반 정책 비교는 추가 label gate를 충족해야 진행 |
| C: H1/H2 | coverage gate와 최대 2분기 실험 | equal-budget에서 추가 근거/회복 성공이 있고 lost successes가 함께 보고됨 |
| D: H3 | fixed-candidate evidence 선택과 별도 source re-read | 검색 추가 효과와 근거 정리 효과를 분리할 수 있음 |
| E: H4/H5 | provenance stream·temporal update·abstention | 단일 M3 이상의 장기 기억 평가와 독립 calibration 자료 확보 |

첫 논문/보고서 수준의 기여 후보는 **멀티모달 상태를 보존하는 로컬 decision policy**, **같은 시간 예산에서 그 절약분을 추가 증거 탐색에 쓰는 검증**, **근거 부족·검색 실패·활용 실패를 구분하는 counterfactual audit**다. 세 항목 모두 현재는 연구 목표다. 충분한 효과가 없으면 “속도 절감은 있었지만 추가 문제 해결은 확인되지 않음” 또는 “중복 prefill로 속도도 개선되지 않음”을 그대로 보고해야 한다.

H4의 장기 기억은 H1–H3의 성과를 설명하기 위한 이름 붙이기가 아니라 별도의 확장 과제다. 정적 corpus의 검색 제어와 실제 시간에 따른 write/revise/prune을 구별해야 기여가 명확해진다.

## 7. 열람·검증 범위

이 문서 작성 중 모델 실행·fine-tuning·외부 decision API 호출은 하지 않았다. 로컬 `agent.py`, `context.py`, G1 summary와 기존 궤적을 읽었고, 위 논문 HTML/초록과 공식 코드·model card·docs를 웹에서 열람했다. 외부 페이지의 벤치마크 수치는 독립 검증하지 않았다. 링크는 본문 주장 가까이에 두었으며 자료를 실행할 권한이나 호환성을 확인했다는 뜻은 아니다.

공유 GPU 실험은 본 문서와 별도 실행 계획을 따라야 한다. 다른 세션의 프로세스를 종료하지 않고, local Gemma 한 인스턴스의 peak resident memory와 실제 가용 자원을 확인한 뒤 시작한다. 이 메모의 분기 개수·학습 단계는 이미 확보된 GPU capacity나 확정 runtime을 전제로 하지 않는다.
