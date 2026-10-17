# 텍스트 통합 action·argument head 관련 연구 업데이트

조사 기준일: **2026-10-17 UTC**. 현재 프로필은 `text_integrated_20261016`, frozen Qwen3-4B-Instruct-2507, keyword_search·semantic_search·read_chunk·FINAL이다. 기존 [novelty review](../lab/reports/text_integrated_20261016/novelty_review_20261016.json)를 최신 공개 연구와 실제 v3 결과에 맞춰 보완한다. 핵심 관련작 중심의 조사이며 전체 문헌을 망라한 systematic review는 아니다. 논문 결과는 저자 보고이며 여기서 재현하지 않았다.

## 가장 가까운 논문

**구조적으로 가장 가까운 것은 LAnR**이다. 같은 LLM hidden state로 검색 벡터와 경량 retrieve/stop head를 만들고 중간 검색어·종료 추론의 텍스트 생성을 줄인다. 따라서 ‘hidden state에서 직접 제어하고 생성 토큰을 줄인다’ 자체를 새 주장으로 삼기 어렵다. [LAnR, 2026-04-20 / v2 05-07](https://arxiv.org/abs/2604.17866v2)

**도구 구성은 GRASP가 가장 가깝다.** semantic search, keyword search, paragraph reading을 조합하며 답변 정확도·근거 읽기·상보 검색·turn 효율을 함께 학습한다. 같은 종류의 세 도구와 종료를 학습한다는 사실만으로 차별화하기 어렵다. [GRASP, 2026-07-11](https://arxiv.org/abs/2607.10463v1)

‘지금 더 생성하면 이후 검색이 줄어드는가’라는 미래 효용 관점에서는 Stop-RAG와 partial-answer utility 연구를 반드시 비교해야 한다. 생성형 결정을 작은 분류기로 대체하는 일반 원리는 TabAgent와도 겹친다. [Stop-RAG](https://arxiv.org/abs/2510.14337), [Partial Answer Quality and Utility](https://arxiv.org/abs/2609.16453), [TabAgent](https://arxiv.org/abs/2602.16429)

## LAnR와 현재 구현의 차이

아래 비교는 저자 방법과 저장된 현재 코드에 근거한 해석이다. LAnR/GRASP는 본문 방법도 확인했고, 나머지 표는 주로 공개 초록·버전 메타데이터를 확인했다.

| 축 | LAnR | 현재 v3 |
|---|---|---|
| 검색 질의 | 지정 `[PRED]` hidden에서 dense vector | 외부 API용 텍스트 인자 생성 |
| 검색기·문서 표현 | 같은 LLM이 문서 표현도 생성 | 별도 고정 embedding 모델·검색 handler 유지 |
| 제어 | retrieve/stop MLP | 4-way action + 5-way read 인자 slot |
| 생성 생략 | 텍스트 검색어 대신 latent retrieval | 관측된 read 인자 후보를 고를 때만 인자 생성 생략 |
| 학습 | 검색 표현·제어 학습을 포함 | frozen backbone, 작은 action/read head만 CE 학습 |
| 현재 증거 | 저자 논문의 실험 | 소규모 파일럿; read 선택 검증 실패 및 후속 탐색 평가 |

LAnR 방법의 핵심은 같은 `[PRED]` 표현에서 질의 벡터와 제어를 얻는 데 있다. 우리 쪽 차이는 기존 keyword/semantic/read 도구 계약을 유지하면서 개별 인자의 생성 여부를 제어하려는 것이다. 이 차이는 설계 범위 차이이며 성능·독창성 입증을 뜻하지 않는다. [LAnR 방법 본문](https://arxiv.org/html/2604.17866v2)

## 직접 관련 연구

날짜는 arXiv 최초 공개일이다. 학회 표기와 출판 완료를 혼동하지 않기 위해 비교는 공개 preprint 기준으로 한다.

| 연구 / 공개일 | 핵심 방법 | 겹치는 부분과 검토할 차이 |
|---|---|---|
| [Stop-RAG](https://arxiv.org/abs/2510.14337), 2025-10-16 | finite-horizon MDP, complete trajectory의 forward-view Q(λ) 목표로 stop/continue 학습 | 남은 검색의 가치 예측 선행. 개별 argument reuse/generate의 비용 대 미래 효용과는 의사결정 단위가 다름 |
| [TabAgent](https://arxiv.org/abs/2602.16429), 2026-02-18 | 실행 trace에서 구조 특징·합성 예제를 만들고 분류기로 routing/shortlisting 등 대체 | 생성형 선택을 classifier로 대체하는 원리 선행. 같은 prefill의 hidden 공유 및 RAG argument 단위 분석을 별도로 보여야 함 |
| [LAnR](https://arxiv.org/abs/2604.17866v2), 2026-04-20 | 같은 LLM의 latent retrieval 및 경량 종료 head | 구조상 최우선 관련작. 외부 검색 API를 유지하는가, 텍스트 인자를 선택적으로 생성하는가가 차이 |
| [TASR](https://arxiv.org/abs/2606.13814), 2026-06-11 | 답변 반복과 calibrated confidence를 활용한 적응 종료 | 저비용 stopping baseline 후보. training-free라는 표현을 calibration·rule 선택에 데이터가 전혀 없다는 뜻으로 읽으면 안 됨 |
| [GRASP](https://arxiv.org/abs/2607.10463v1), 2026-07-11 | 세 검색/읽기 도구와 정확도·근거·turn 효율 보상으로 RL | 인터페이스가 가장 유사. 우리 head의 imitation 학습 및 인자 생성 생략과 RL 전체 정책 학습을 구분해야 함 |
| [MetaRAG](https://arxiv.org/abs/2608.24214), 2026-08-25 | Verify-first 행동 생성, 내부 answerability belief와 correctness-gated 일관성 보상 | belief head가 있다는 점만 같다고 동일 방법은 아님. 저자 probe는 학습 때만 쓰며 inference 제어 head와 역할이 다름 |
| [AnySearch / One Policy, Any Budget](https://arxiv.org/abs/2609.00813), 2026-09-01 | budget scaffold 이후 scaffold를 제거한 curriculum RL, 정확도와 예산 효율 결합 | step·budget 최적화 자체도 선행. 우리 현재 실험은 RL·budget conditioning을 하지 않았음 |
| [Better Together](https://arxiv.org/abs/2609.05637), 2026-09-04 | 강한 dense/rerank/MMR 기준에서 질의 rewrite와 보완적 fusion, confidence 기반 선택적 rewrite 분석 | 새 검색어를 생성할지 선택한다는 방향과 겹침. simulated router 분석을 실제 온라인 latency 검증으로 오해하지 말 것 |
| [Predicting Partial Answer Quality and Utility](https://arxiv.org/abs/2609.16453), 2026-09-15 | 반복 중간 답변의 현재 품질과 추가 검색의 marginal utility를 구분해 예측 | 현재 confidence와 미래 개선량은 다른 목표. argument 생성 가치 역시 미래 비용·품질 target이 필요하다는 근거 |

GRASP의 본문은 GRPO와 답변·근거 읽기·검색·turn 효율 보상을 사용하고 생성형 reasoning/action을 포함한다. 현재 방식은 검색 인자를 생성하고 read에서만 기존 후보 선택이 가능하며 backbone을 RL로 바꾸지 않는다. 그러므로 GRASP 점수와 현재20문항 점수를 직접 비교할 수 없다. [GRASP 방법 본문](https://arxiv.org/html/2607.10463v1)

## 2026년 10월 최신 관련 연구

| 연구 / 최초 공개 | 확인된 내용 | 이번 연구에 주는 시사점 |
|---|---|---|
| [Agentic RAG Evaluation: Budget Allocation Across Questions, Trajectories, and Reads](https://arxiv.org/abs/2610.05034), 10-04 | HotpotQA/MuSiQue에서 질문 수·trajectory·반복 read에 평가 예산을 배분하고 비교 정밀도 분석 | 작은20문항의 동률·부분 개선을 일반화 증거로 확대 해석하면 안 됨. 다음 확대에서 질문 다양성 우선은 이 결과에 근거한 제안이며 모든 설정의 정리는 아님 |
| [Beyond Semantic Similarity: Performance and Costs of Agentic Retrieval for Complex Tasks](https://arxiv.org/abs/2610.05750), 10-05 | ReAct 검색이 같은 embedder의 일반 검색보다 검색 품질을 높이지만 큰 입력 토큰·시간 비용을 보고 | 출력 토큰만으로 효율을 평가할 수 없음. 전체 입력·도구 호출·episode wall time을 함께 보고해야 함. 모델/과제/장비가 달라 논문 시간과 우리 시간을 직접 비교하지 않음 |
| [Frustratingly Easy Agentic RAG via Portable Skills](https://arxiv.org/abs/2610.14428), 10-10 | 고객지원에서 portable instructions·경량 도구·탐색용 corpus로 단순한 agent 구성, 사람 검토 기반 평가 | 구성 단순화가 자동으로 빠른 시스템을 뜻하지 않음. 고객지원 절차 평가와 MuSiQue 답변 F1은 서로 다른 목표 |

## 독창성 판단과 다음 증거

현재 다음 주장은 선행 연구와 겹친다: hidden-state 제어 head, 생성형 tool 선택의 classifier 대체, 적응 종료, 효율 보상, 선택적 질의 rewrite. ‘최초’라는 표현을 쓰지 않는다.

구체적으로 남겨 둘 가설은 **같은 도구 계약과 관측에서, 인자를 바로 선택할지 새로 생성할지 결정하면서 현재 생성 비용과 이후 남은 interaction 비용을 함께 예측하는 것**이다. 이는 [KEEP 가설 기록](../lab/registry/text_integrated_20261016/KEEP_generation_value_vs_remaining_steps.json)의 연구 제안이다. 현재의 teacher exact-match 기반 slot CE는 이 미래 가치를 학습하지 않는다. 관련 문헌을 더 확인하고 실제 반사실 비교·cost-to-go supervision의 가치를 보여야 하며, 이 문서만으로 새로운 기여라고 결론내리지 않는다.

검토 순서는 동일 checkpoint의 argument 선택 OFF/ON, 생성 없이 관측 후보를 고르는 단순 heuristic, 적응 종료 baseline, 이후 필요 시 cost-to-go 목표의 기여 분리다. LAnR는 문서 표현·인덱스까지 바뀌므로 기존 API 내부의 작은 교체처럼 취급하면 공정하지 않다. 이 순서는 후속 연구 제안이며 추가 학습/RL 실행 승인이 아니다.

## 현재 실험과 연결

기존 평가20은 native 답변 F1 29.00% / EM25.00%, action head + always-generate F1 30.43% / EM25.00%였다. 생성 토큰 평균89.75→60.55지만 강제 prefix를 합하면89.75→91.10이고 입력 토큰도3928.5→4266으로 늘었다. 도구 step도2.20→2.35이다. 전체 비용·step 감소 또는 품질 향상을 입증한 결과가 아니다. [원 결과](../lab/reports/text_integrated_20261016/improvement_v3_final.md)

read 선택은 검증8에서 F1 21.88%→9.38%, EM12.5%→0%로 떨어져 당시 primary에서 제외했다. 사용자 요청으로 같은 기존20문항에서 **선택 ON/OFF를 모두 평가하는 후속 탐색 비교**를 완료했다. ON은 read15회 중8회 생성을 생략했지만 F1 30.43%→25.43%, EM25%→20%로 품질이 떨어졌다. 이 비교는 원 primary를 바꾸거나 새 heldout이라고 부르지 않는다. 결과·선택 횟수·인자 토큰·step·latency는 별도 보고서 `lab/reports/text_integrated_20261016/improvement_v3_argument_selection.md`에 기록했다.

episode latency는 제어·인자·답변 생성·검색(query embedding 포함)·Python 처리를 포함한다. 모델 로딩·corpus 사전 embedding·채점·저장·백업은 제외한다. 공유 GPU 관측값을 통제된 speedup으로 해석하지 않는다. [step·latency 정의 및 집계](../lab/reports/text_integrated_20261016/improvement_v3_steps_latency.md)

## 딥서치 후속 보완 — 2026-10-17

LAnR 부록B.1의 실제 control head는 단일affine+sigmoid이며, 본문의MLP라는표현보다구체적이다. [개별정리](papers/LANR_2604_17866_20261017.md). Frozen host의작은router에후보평가·품질비용학습을적용한FORGE도직접관련작으로추가했다. Full과Lite의추가host probe비용을구별해야한다. [FORGE개별정리](papers/FORGE_2609_34358_20261017.md). 최신종합판단은Astra딥서치문서에서보완하며이전조사의완전성을주장하지않는다.
