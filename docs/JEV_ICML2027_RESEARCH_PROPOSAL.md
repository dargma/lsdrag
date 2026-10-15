# jev_v1.1 — ICML 2027 연구 방향 조사와 제안

작성: 2026-10-15 UTC. 상태: **문헌 조사와 미검증 연구 제안**. 새 GPU 실험 결과가 아니다. 사용자가 요청한 세 에이전트의 병렬 조사(decision/control, multimodal, memory)를 종합하고, 별도 Astra 모델의 비판·정교화를 반영했다.

**최신 판단:** §4–5의 초기 '두 단계 value head'는 독창적 방법으로 채택하지 않는다. 기존 Q/value 학습과의 차이가 부족하다. 현재 추천은 §12의 **query generator와 final reader에 대한 증거 접근을 교차 개입해 제어 학습 신호를 만드는 조건부 pilot**이다. 논문 주제를 확정한 것이 아니며, [Astra 검토 원문](JEV_ICML2027_ASTRA_REVIEW.md)의 판정·조건을 함께 읽는다.

## 1. 사용자 목표와 제약

Agentic RAG의 각 iteration에서 tool call·답변·종료를 결정하는 데 쓰이는 autoregressive generation을 최소화하고, 상태에서 선택지별 점수/확률을 직접 계산해 필요한 행동을 빠르게 선택한다. 검색어·도구 인자·최종 답변처럼 내용을 구성해야 하는 작업에는 생성을 사용한다.

목표는 2027년 1월 ICML 제출을 준비하는 것이다. 정확한 ICML 2027 제출 마감은 이번에 확인하지 못했다. [공식 future meetings](https://icml.cc/Conferences/FutureMeetings)와 향후 CFP를 확인해야 하며, 1월은 작업 일정의 가정이다. Blackwell GPU 한 대를 사용할 수 있으나 정확한 모델·VRAM·사용 시간은 미확인이다.

멀티모달 임베더와 Reader VLM을 사용하는 방향에 관심이 있다. RAG 효율, memory, CoT를 모두 구현하는 framework를 원하지 않으며, 핵심 문제 하나를 선정한다. memory와 CoT는 중심 주제 후보이지 반드시 포함할 구성요소가 아니다.

## 2. 현재 실험의 정확한 위치

현재 코드/결과는 [PoC 보고서](../lab/reports/jev_v1.0/PROGRESS.md), [학습 감사](../lab/reports/jev_v1.0/HEAD_TRAINING_AUDIT.md), [기존 계획](JEV_EXPERIMENT_PLAN.md)에 보존되어 있다.

- 같은 Gemma 4 E4B-it backbone을 동결하고 마지막 정규화 hidden 2,560차원에서 Linear(2560,4), 10,244개 파라미터로 keyword_search / semantic_search / read_chunk / FINAL을 선택했다.
- 서로 다른 성공 질문 45개·46궤적·197턴으로 한 번 학습했다. 160/197턴은 텍스트 MultiHop-RAG이며 이미지가 전달된 상태는 12/197턴이다.
- 새 M3 20문항에서 평균 전체 시간 7.273→5.048초(30.6% 감소), EM 1→2문항. 유일한 EM 개선은 기존 답도 의미상 맞았던 출력 형식 차이다.
- 전체 호출은 52→54회로 늘었다. prefix·출력 길이·탐색 경로가 함께 달라져 head 연산만의 인과적 속도 효과를 분리하지 못했다.
- 학습 행동 일치율 197/197은 일반화 근거가 아니다. FINAL softmax는 정답 확률이 아니다.

따라서 현재 결과는 **실제 멀티모달 Reader에서 행동 선택을 생성과 분리할 수 있다는 구현 PoC**다. 새로운 추론 능력, 정확도 개선, 충분한 멀티모달 정책 학습, 보정된 종료 확률을 입증한 결과는 아니다. Jev의 비공개 내부 구조를 재현했다고 부르지 않는다.

## 3. 핵심 문헌과 독창성 충돌

아래는 원문 방법·평가를 대조한 핵심 자료다. arXiv 초고는 동료심사를 통과한 확정 결과와 구분하며, 저자 보고를 이 저장소의 재현 결과처럼 취급하지 않는다. 조사 범위 밖 선행연구가 없다는 보장은 없다.

| 문헌 | 확인한 내용 | 우리 주제에 대한 제약 |
|---|---|---|
| [TypeSafe Jev 공식 문서](https://docs.typesafe.ai/introduction) | typed choice/score 등 bounded decisions. 자세한 내부 architecture·학습법은 공개 설명만으로 재현 불가 | Jev에서 영감을 받은 로컬 controller와 실제 Jev를 구분 |
| [General Decision Models / InnerJev](https://arxiv.org/html/2610.03935v1), 2026-10 | first-token readout, reasoning-to-readout self-distillation. §5.3의 agent 실험에서 typical latency 감소와 성공률 하락·tail 증가를 함께 보고 | head·첫 토큰·CoT 증류 자체를 새 기여로 내세우기 어려움. 표와 본문 수치 일부 불일치도 있어 상세 숫자는 원문 버전 재확인 필요 |
| [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/) | 작은 classifier로 질문별 retrieval strategy 결정 | 작은 router로 RAG 효율화 자체는 선행 |
| [Probing-RAG](https://arxiv.org/html/2410.13339v1) | hidden-state prober로 추가 검색 판단. rationale/answer 생성 뒤 상태를 읽음 | hidden-state classifier 자체는 선행. 생성 전 제어와의 차이를 명시해야 함 |
| [Stop-RAG](https://arxiv.org/html/2510.14337v1) | finite-horizon MDP, Q(lambda) 기반 STOP/CONT, 미래 return 학습 | value-based stopping·비근시안적 제어 자체는 선행. 이 방법을 단순한 immediate-gain baseline으로 묘사하면 안 됨 |
| [Calibration Is Not Control](https://arxiv.org/html/2606.21399v1) | failure confidence와 intervention advantage 구분, 같은 prefix의 branch outcomes로 제어 평가 | confidence 대신 utility 학습, counterfactual rollout 일반론도 선행 |
| [Calibrate-Then-Act](https://arxiv.org/html/2602.16699v1) | cost–uncertainty tradeoff를 명시해 sequential exploration 제어 | cost-aware exploration 자체는 새 개념이 아님 |
| [UniversalRAG](https://aclanthology.org/2026.acl-long.177/) | modality·granularity routing으로 heterogeneous corpus 검색 | 멀티모달 임베더+router라는 조합만으로 부족 |
| [Learning to Search](https://arxiv.org/html/2604.07146v1) | 생성형 reasoning 뒤 Answer/Text/Image/Caption 행동 선택 | 멀티모달의 순차 행동·도구 선택 자체는 선행 |
| [MC-Search / Search-Align](https://arxiv.org/html/2603.00873v1) | cross-modal multihop, 단계별 modality·근거·중간 답 annotation, process-supervised agent. §2.1에 context utility와 navigational role Nav(t)도 정의 | 좋은 task/process baseline. 연속 멀티모달 탐색이나 navigation 가치 자체를 새 발견으로 주장하면 안 됨. Nav(t)의 entity proxy와 개입 기반 결과 차이는 구별해야 함 |
| [CLIMB](https://arxiv.org/html/2610.03421v1) | complementary pool + confidence-controlled refinement/stopping | 상보적 증거와 종료를 결합했다는 넓은 주장도 충돌 |
| [Beyond the Query](https://arxiv.org/html/2609.12437v1) | matched query-only vs query+retrieval router에서 tested state features의 추가 효과가 확실하지 않았음 | 현재 state를 넣었다는 사실만으로 adaptive 기여를 인정하지 않음 |
| [Jev-Mem](https://arxiv.org/html/2609.23986v1) | memory typing·routing·budget·traversal·scoring·stopping에 decision control | memory lifecycle에 Jev를 적용한다는 방향은 직접 중복 |
| [AgeMem](https://arxiv.org/html/2601.01885v1), [TrustMem](https://arxiv.org/html/2606.25161v1), [MemGate](https://arxiv.org/html/2606.06054v1), [JitMem](https://arxiv.org/html/2609.27334v1) | memory action 학습, 전이 검증, 작은 admission gate, read-time curation 등 | memory로 분야를 옮기는 것만으로 새로움이 생기지 않음 |

## 4. 초기 방향 비교 — Astra 검토 전 후보 기록

| 후보 | 이점 | 가장 큰 위험 | 잠정 판단 |
|---|---|---|---|
| 일반 Agentic RAG를 decision head로 가속 | 기존 구현 재사용, 비용 측정 쉬움 | first-token/기존 value control과 중복 | 필수 baseline/PoC로 유지 |
| memory 전체 framework | 장기 상호작용 문제와 연결 | Jev-Mem과 직접 중복, 새 데이터·평가·시스템 범위 과다 | 주력으로 추천하지 않음 |
| CoT를 빠른 readout으로 증류 | 사용자 목표와 직접 연결 | InnerJev·adaptive thinking 연구와 중복 | 학습 도구/비교군 후보 |
| cross-modal 연속 정보 획득의 지연된 가치를 학습하는 빠른 제어 | 멀티모달이 필요한 이유와 decision의 실패를 함께 검증 | 현상이 드물거나 기존 multi-action value learner로 충분할 수 있음 | **2주 반증 실험을 우선할 주력 후보** |

이 추천은 새로움·성능이 검증된 결론이 아니다. 다음 가설과 실험을 통과해야 주제로 확정한다.

## 5. 초기 후보: 멀티모달 증거의 지연된 가치를 고려하는 비생성 제어

가제: **Fast Decisions Beyond Immediate Evidence: Non-Generative Control for Cross-Modal Retrieval**.

연구 질문:

> 한 번의 검색으로 답이 바로 좋아지지 않더라도, 서로 다른 종류의 증거를 연속해서 얻으면 풀리는 문제에서, 작은 비생성 controller가 너무 이른 종료를 피하며 생성형 planner의 제어 비용을 줄일 수 있는가?

예를 들어 사진에서 대상을 식별한 뒤 그 대상의 역사적 속성을 텍스트로 찾아야 한다. 첫 이미지 탐색은 최종 답을 직접 주지 않지만 다음 검색을 가능하게 한다. 단, 이것은 설명용 예시이며 현재 PoC에서 이 실패가 얼마나 발생했는지는 측정하지 않았다.

기여 후보는 '멀티모달 RAG'나 '두 단계 계획' 자체가 아니다. **어떤 상태에서 즉시 효용을 학습한 빠른 controller가 실패하는지 측정하고, 그 실패에 필요한 짧은 미래 상호작용을 작은 모델에 학습시킬 수 있는지**다. 일반적 상보성·MDP·Q-learning을 새 이론으로 소개하지 않는다.

### 5.1 최소 방법 초안

1. Frozen multimodal retriever와 frozen Reader VLM, 동일한 corpus·이미지 전달 정책을 사용한다.
2. 최소 행동은 ANSWER/STOP, TEXT_SEARCH, IMAGE_SEARCH로 둔다. 현재 A-RAG의 keyword/semantic/read 도구와 일대일 대응하지 않으므로 새 인터페이스는 모든 대조군에 공통 적용한다. A-RAG 원형 성능과 공통 인터페이스 성능을 구분한다.
3. 현재 질문·관측된 증거·이력의 표현에서 작은 controller가 행동 점수를 읽는다. 검색어/인자 생성과 최종 답은 공통 VLM에 맡긴다.
4. 학습 시 같은 상태에서 stop, 한 단계 획득, 제한된 두 단계 획득을 실행해 answer quality와 전체 비용을 기록한다. 두 번째 행동의 검색어는 첫 관측을 받은 뒤 생성한다. 미래 gold entity나 gold query를 미리 주지 않는다.
5. 즉시 value와 짧은 후속 경로의 가치를 비교하도록 학습하는 방법을 검토한다. 구체 loss·architecture는 아직 확정하지 않는다. 단순 horizon 증가나 더 많은 label의 효과와 분리할 수 있어야 한다.
6. 배포 시 실제 미래 branch를 모두 실행하는 planner로 바꾸지 않는다. 현재 관측만으로 행동을 선택하고 매 iteration 갱신한다. 미래 branch는 학습·진단용으로만 사용한다.

설명용 표기: 현재 상태 s에서 지금 답하는 품질을 U(s), 실제 획득 a 뒤 상태를 s_a라 하자. 단일 관측 이득 U(s_a)-U(s)는 작지만, 조건부 후속 획득 b 뒤 U(s_ab)-U(s)는 큰 상태를 찾는다. 비용을 함께 빼서 평가한다. 단일 샘플의 성공 차이를 기대 효용의 정확한 추정이나 인과적 증명으로 취급하지 않는다. 또한 이득 부재가 Reader 약점 때문인지, 실제 정보 부재 때문인지 구분한다.

### 5.2 차별성을 반증하는 가장 강한 대조

Stop-RAG는 이미 미래 가치를 학습한다. 따라서 one-step classifier만 이기는 결과로는 부족하다. **같은 행동 집합·같은 상태·같은 rollout label을 받은 표준 multi-action Q/value learner**가 같은 결과를 내면, 제안된 특수 방법의 필요성이 약해진다.

상보성의 추가 학습 objective나 표현을 제안한다면 동일 데이터·horizon·파라미터/연산 예산 조건에서 generic value learner보다 좋아야 한다. 단순 imitation 대비 향상만으로 새 알고리즘을 주장하지 않는다.

## 6. 필수 baseline과 공정성

| 비교 | 검증 목적 |
|---|---|
| 동일 VLM의 기존 생성형 controller | 원래 방식 대비 시스템 결과 |
| 동일 VLM의 constrained one-token action logits | 긴 JSON/CoT를 없앤 당연한 효과와 구분 |
| 같은 state·data의 linear/MLP imitation head | head architecture와 outcome supervision의 효과 분리 |
| immediate-value controller | 즉시 answer gain의 한계 검사 |
| 같은 branch data의 표준 multi-action future-value learner | 기존 비근시안적 제어 대비 제안 기여 |
| query-only controller | state-conditioned 정보의 실제 추가 가치 |
| fixed retrieval budget, always-text/always-image/공통 혼합 정책 | 정책 복잡도가 실제 필요한지 검사 |
| 관련 공개 multimodal agent/UniversalRAG/CLIMB 중 task-compatible한 방법 | 외부 방법 대비 위치. 원형과 이식 구현을 분명히 표시 |

주 비교는 같은 reader·retriever·corpus·context/image budget·query generator·최종 출력 형식·backend로 controller만 바꾼다. One-token baseline도 실제 문맥에서 label tokenization과 tool/argument 연결 비용을 확인한다. 미래 라벨을 받은 baseline과 받지 않은 baseline을 섞어 algorithm 효과로 해석하지 않는다. 전 조건에 같은 answer prefix를 사용해 기존 PoC의 형식 confound를 막는다.

## 7. 데이터와 평가

- 첫 개발은 기존 M3 자료와 [CrossModalQA](https://arxiv.org/html/2609.05518v1)의 200–300문항 정도로 예상한다. CrossModalQA는 1,863 QA·4,987 articles·4,431 images로 보고되어 상대적으로 작다. 실제 다운로드·라이선스·분할·정답 근거 접근 가능성은 아직 확인/재현하지 않았다.
- [MC-Search](https://arxiv.org/html/2603.00873v1)는 modality/chain annotation과 Search-Align baseline이 장점이다. corpus가 커 준비 비용을 측정한 뒤 본 실험 범위를 정한다. gold chain을 제공한 진단은 실검색 평가와 분리한다.
- 외부 검증 후보는 [VisDocAgentBench](https://arxiv.org/html/2608.17889v1). 작은 test-only benchmark를 학습·threshold 선택에 쓰지 않는다.
- [LongMemEval-V2](https://xiaowu0162.github.io/longmemeval-v2/)는 memory로 확장할 경우의 후속 후보다. 이번 주력 실험에 추가하지 않는다.

주 지표는 **최종 QA 품질–전체 wall-clock Pareto**다. 결정 정확도·토큰 절감만 보고하지 않는다. 평균·p50·p95 지연, prefill/visual encoding/control/query generation/retrieval/answer/fallback 시간을 계측하고 이미지 토큰·검색/읽기 횟수를 기록한다. 학습 라벨 수집 비용은 배포 지연과 별도 보고하되 숨기지 않는다.

process 진단은 성급한 종료, 완전한 evidence-chain coverage, controller가 선택한 branch의 관측 regret, single-step 대비 pair-gain 상태의 빈도다. 사후 가장 좋은 branch를 택한 oracle은 upper bound 진단이지 배포 성능이 아니다. noisy realized best-of branch를 정확한 최적 정책으로 부르지 않는다.

질문뿐 아니라 가능한 경우 entity/document/source 기준으로 분리한다. 같은 원본에서 나온 모든 state·반사실 쌍은 같은 split에 둔다. 문항 단위 paired bootstrap과 seed 반복을 사용한다. 품질 유지의 허용 손실 폭·통계적 판단은 본 평가 전에 정하며, 단순 p>0.05를 비열등성 증명으로 쓰지 않는다.

Reader가 충분한 gold evidence를 받아도 거의 못 푸는 환경에서는 controller의 개선 여지가 없다. oracle-evidence Reader 진단을 먼저 수행하고 기존 M3의 낮은 EM을 controller 문제로 단정하지 않는다.

## 8. 2주 반증 pilot 및 중단 조건

1. 정확한 GPU·VRAM·가용 시간 확인, 30–50문항에서 runtime/메모리/branch 비용 측정.
2. 결과를 보기 전에 200–300문항 pilot split과 비용/품질 평가를 고정한다. 축소 corpus면 distractor 포함과 범위 한계를 명시한다.
3. 같은 prefix에서 stop / text / image / text→image / image→text를 제한적으로 분기하고, pair benefit과 조기 종료를 측정한다. 실행 결과에 따른 후속 query 작성은 허용하되 gold는 입력하지 않는다.
4. 즉시/미래 value learner, one-token readout, 작은 제안 controller를 최소 비교한다. 쉬운 성공 궤적만 수집하는 현재 PoC 방식과 달리 실패·복구 불가 상태도 포함한다.
5. 현상이 충분한지, oracle 여지가 실제로 있는지, 현재 state로 이득을 예측할 수 있는지, generic value baseline 이상의 개선이 있는지 순서대로 판단한다.

**중단/재설계:** pair benefit이 드묾; full evidence에서도 Reader가 못 품; query-only가 같음; generic future-value learner가 충분함; 전체 prefill/인자 비용을 포함하면 속도 이득이 사라짐; 새 문서에서 효과가 사라짐. 이런 경우 새 이름을 붙여 규모를 늘리지 않는다.

동일 품질에서 전체 지연 20% 이상 절감 같은 수치는 pilot go/no-go의 예시일 뿐 예상 결과가 아니다. 실제 기준은 작은 runtime 확인과 품질 손실 허용 폭을 근거로 본 평가 전에 확정한다.

## 9. GPU 한 대와 제출 일정

Frozen 3B–8B급 reader의 inference, cached features와 작은 head 학습을 출발점으로 한다. 실제 fit 여부는 VRAM·이미지 수·context 길이에 달려 있다. Full agent RL, retriever+VLM 공동 학습, memory framework 동시 구축은 현재 일정의 기본안에 포함하지 않는다. LoRA는 frozen 표현의 한계를 확인했을 때만 검토한다.

가장 큰 비용은 head 학습보다 **branch trajectory 수집과 반복 평가**일 가능성이 있다. 먼저 수십 문항의 실제 GPU-hour를 재고, 전체 수집량·seed·핵심 baseline 수를 산정한다. 모델의 크기만으로 완료 시간을 약속하지 않는다.

| 기간(잠정) | 산출물과 판단 |
|---|---|
| 10월 후반 | 현상·독창성·oracle 여지의 2주 pilot, 주제 확정 또는 중단 |
| 11월 | 최소 방법, matched baselines, 주 데이터 학습/개발 |
| 12월 전반 | held-out 평가, 핵심 ablation, 다른 reader/외부 데이터의 핵심 비교 |
| 12월 후반 | 통계·실패 분석·재현 정리, 논문 초안 |
| 1월 | 공식 마감에 맞춘 최종 검증·집필. 새 범위 확장은 제한 |

## 10. ICML 2027 전망: 사실과 추론

사실: [ICML 2026 CFP](https://icml.cc/Conferences/2026/CallForPapers)는 decision/control, ML systems, evaluation, probabilistic methods 등을 폭넓게 포함한다. [AdaMEM](https://proceedings.mlr.press/v306/zhang26cn.html) 같은 memory 연구와 [When to Think, When to Speak](https://proceedings.mlr.press/v306/wei26i.html) 같은 reasoning/latency 연구도 2026 proceedings에 있다.

**추론:** 2026 하반기 논문과의 경쟁을 고려하면 2027에는 새로운 framework 이름보다, 명확한 학습 문제·실패 원인·공정한 compute 비교·일반화가 더 설득력 있을 것으로 예상한다. 특정 키워드의 유행이나 ICML 채택 가능성을 예측/보장하는 것은 아니다. RAG라는 단어의 연식만으로 연구 가치를 판단할 근거도 없다.

잠정 논문 메시지는 **'빠른 결정을 위한 작은 모델이 지연된 정보 가치를 놓치는 조건을 밝히고, 그 가치를 학습해 생성형 제어의 비용을 줄인다'**다. 멀티모달은 이 질문에 필요한 실험 구조이며, memory·CoT를 모두 붙이는 확장 계획이 아니다.

## 11. 대안 하나와 미결정 사항

주력 현상이 약할 경우 대안은 **값싼 text/OCR 관측에서 비싼 image/high-resolution 관측으로 넘어갈 시점**만을 학습하는 것이다. relevance와 actual visual-read benefit을 구분하고 같은 state에서 read/skip outcomes를 비교한다. 다만 이 또한 UniversalRAG·visual perception scheduling·value-of-information과 겹치므로 독창성을 별도 감사해야 한다. 주력과 동시에 구현하지 않는다.

미결정: 정확한 GPU/가용 시간, corpus 공개 자산의 접근 가능성, 주 reader의 evidence-conditioned 능력, pair-benefit 현상의 빈도, baseline 대비 필요한 특수 objective, 공식 ICML 2027 일정. **이 문서는 연구 실행을 완료했다고 주장하지 않으며, 새 실험은 아직 시작하지 않았다.**

## 12. Astra 검토 후 정교화 — 증거가 다음 검색에 주는 영향과 답변에 주는 영향

초기 제안의 generic two-step value/Q 학습은 기존 방법과 구분되지 않아 **새 방법으로는 기각**한다. 지연된 효과를 확인하는 현상 pilot은 조건부로 유지한다. 더 좁은 질문은 다음과 같다.

> 현재 얻은 증거가 최종 답변을 직접 개선하지 않아도 다음 검색을 가능하게 할 수 있다. Query generator에 대한 증거 접근 효과와 final reader에 대한 접근 효과를 따로 측정한 학습 신호가, 같은 rollout 예산의 일반 value 학습보다 빠른 제어의 일반화에 도움이 되는가?

현재 관측된 증거 e를 query generator에 보임/숨김, final reader에 보임/숨김의 2×2 module-access intervention을 만든다. 선택 대상 e는 gold 정답이나 gold chain으로 고르지 않고 사전에 정한 관측 기준으로 정한다. Query generator가 새 검색을 실행하고 final reader가 답하는 과정에서 네 조건의 downstream quality·cost 차이를 기록한다. 이는 새로운 증거를 gold로 주입하는 방식이 아니다.

두 모듈에 대한 접근을 바꾼 효과는 해당 구현 안에서의 **module-access intervention effect**다. Query가 e의 정보를 복사하거나 새 검색 결과가 같은 내용을 되돌려줄 수 있으므로, 이를 인간적 의미의 '순수 navigation 효과'나 자연적 직접/간접 인과효과의 식별이라고 부르지 않는다. Query/history 직접 전달과 검색 결과를 통한 재노출을 감사해야 한다. MC-Search의 Nav(t), CUE-R의 evidence perturbation, VAD의 evidence-present/removed distillation과도 별도 대조한다.

새 방법 후보는 이 paired supervision을 이용해 현재 상태의 비생성 readout을 학습하는 것이다. 차별성은 2×2 표나 small head 자체가 아니라, **일반 outcome label로 학습할 때 잃는 구분을 이 신호가 실제로 회복하는지**에 달려 있다. Same-data·same-budget generic Q/value baseline, answer-only evidence removal label, MC-Search Nav proxy, 단순 auxiliary loss를 반드시 대조한다. 세부 estimator·loss·실행 절차와 반증 기준은 [Astra 검토](JEV_ICML2027_ASTRA_REVIEW.md)에 기술한다.

선행연구 추가: [CUE-R](https://arxiv.org/html/2604.05467v1)는 evidence intervention과 두 support의 비가산 효과를 다룬다. [VAD](https://arxiv.org/html/2607.28590v1)는 evidence present/removed 차이로 distillation target을 구성한다. [Active Acquisition for Multimodal Temporal Data](https://arxiv.org/abs/2211.05039)는 modality acquisition의 비용–예측 성능 tradeoff를 학습한다. 따라서 intervention, 상보성, acquisition, distillation 각각을 새 개념으로 주장하지 않는다.

우선 확인할 것은 네 조건의 차이가 측정 잡음보다 큰지, 접근 효과를 현재 관측만으로 예측할 수 있는지, 이 신호를 학습에 넣었을 때 독립 문서/대상에서 제어 결과가 개선되는지다. 이 세 조건이나 전체 지연의 이득이 성립하지 않으면 주제를 확정하지 않는다. Memory lifecycle이나 긴 CoT 학습을 추가해서 이 실패를 덮지 않는다.
