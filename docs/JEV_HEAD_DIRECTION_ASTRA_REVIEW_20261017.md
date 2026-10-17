# Astra 방향 검토: head 정교화, 인자 제어, interaction 압축

2026-10-17 UTC. 사용자의 다방향 검토 요청에 대한 연구 판단이다. 저장 결과와 코드, 1차 논문을 읽었으며 새 GPU 추론·학습·실험은 실행하지 않았다. 기존 split, checkpoint, primary와 실패 결과를 보존한다. 아래 수치 gate는 **다음 파일럿의 제안 기준**이며 이미 통과한 기준이나 통계적 유의성 기준이 아니다.

**추천은 A를 기준선으로 유지하면서, 대규모 일반화 평가 전에 C의 압축 가능성을 작은 반사실 실험으로 확인하는 것이다.** B는 자유형 인자 생성 head로 확장하지 않고, 관측 후보 선택과 새 생성의 가치 차이를 나중에 C에 연결한다. A/B/C를 동시에 새로 학습하면 지금의 자료 규모에서는 무엇이 기여했는지 구분하기 어렵다. 2-hop100/3-hop30 계획은 폐기할 필요는 없지만, 현재 그대로 즉시 시작하는 데에는 반대한다.

## 1. 현재 증거가 말하는 것

| 실제 결과 | 해석 가능한 범위 |
|---|---|
| native F1 .2900 → action head + generated args .304286, EM .25 → .25; 20문항 중 부분점수 1승/19동률 | 작동하는 작은 파일럿이다. 품질 개선·동등성·일반화 입증은 아니다. |
| tool calls 2.20 → 2.35; generated tokens 89.75 → 60.55; 강제 prefix 포함 89.75 → 91.10; input tokens 3928.5 → 4266 | 출력 생성 일부를 일괄 prefix 처리로 바꾼 효과가 있다. 전체 FLOPs나 latency가 토큰 합에 비례하지 않으므로, 합계 증가가 느려졌다는 증명도 아니다. |
| 같은 head의 read 선택 ON: F1 .304286 → .254286, EM .25 → .20, 8/15 read 생성 생략 | 인자 생성 호출 절약은 확인했지만 품질 유지 실패. 평균 인자 절감은 3.4 tokens이고 tool steps는 그대로다. |
| 고정 validation 29상태 action macro-F1 .814, premature FINAL 0 | 자연 teacher 상태에서 imitation 신호를 학습했다. 새로운 자율 상태에서 최적 행동을 고른다는 증거는 아니다. |
| train label의 turn별 최빈 행동 baseline: validation macro-F1 .609, premature FINAL 5 | head가 이 간단한 순서 baseline보다 낫다. 다만 순서/길이 shortcut을 완전히 배제한 것은 아니다. |

원자료: [v3 최종 결과](../lab/reports/text_integrated_20261016/improvement_v3_final.json), [인자 선택 ON/OFF](../lab/reports/text_integrated_20261016/improvement_v3_argument_selection.json), [CPU 구조 audit](../lab/reports/text_integrated_20261016/head_direction_structural_audit_20261017.json).

teacher32는 111상태이지만 독립 질문은 32개이고, 학습은 24질문/82상태다. 모든 teacher trajectory는 tool 2회 또는 3회뿐이다. 따라서 현재는 긴 interaction 압축을 학습할 자료가 없다. 57회 search 중 12회가 빈 결과이고 22회 read 중 5회는 반환 전문 전체가 이전 snippet과 동일했다. 이들은 조사할 단서다. 빈 검색도 실패 정보를 제공하며, 중복 read를 지우면 이후 prompt와 정책 상태가 바뀌므로 안전 생략의 증거로 세면 안 된다.

read-slot 실패는 후보 부족 하나로 설명되지 않는다. validation 6 read 상태 중 teacher 인자가 후보로 표현 가능한 것은 5개인데 slot 정확도는 2/6이다. train의 slot4 양성은 0이고, teacher exact match도 downstream utility와 다르다. 표본·표현·후보 계약·label 목표를 함께 봐야 한다. ON/OFF에서 인자가 달랐지만 답은 맞게 유지된 경우도 있어, slot imitation 정확도만 최대화하는 것이 QA 최적화와 같지 않다.

현재 “통합학습”도 정확히 해석해야 한다. frozen feature 위의 action4/read5 독립 linear head를 같은 batch/optimizer에서 CE 합으로 학습하며, trainable shared trunk는 없다. 따라서 read loss가 공유 표현을 개선해 action 학습을 도왔다는 multitask transfer는 현 구조에서 주장할 수 없다. 공유 prefill·checkpoint·실행 계약의 통합과 공유 표현 학습은 다르다.

## 2. 세 방향에 대한 공정한 판단

| 방향 | 과학적 질문과 실용 가치 | 핵심 위험 / 데이터 요구 | 지금의 판단 |
|---|---|---|---|
| A. action head 일반화·정교화 | 같은 관측에서 LLM의 생성 결정을 작은 readout으로 대체해 품질·비용을 유지하는가? 적은 질문으로 학습한 정책이 hop/실패 상태 변화에 견디는가? | teacher 행동이 정답은 아님. 상태 수를 질문 수로 착각하기 쉬움. tool 순서 shortcut, 자율 rollout shift, confidence 과신. | 가장 실행하기 쉬운 기반. 기존 head 유지; architecture sweep보다 원인 분리와 강한 baseline이 먼저. |
| B1. 관측 인자 pointer/집합 선택 + GENERATE gate | 이미 관측한 ID·값을 고르는 비용과 새 값을 생성해 얻는 미래 효용의 차이를 학습할 수 있는가? | 후보 밖 값은 생성해야 함. 정적 slot 순서는 취약. 후보 내용·복수 ID·coverage·반사실 품질 label 필요. | 제한적 후속 후보. 기존 read 선택 실패를 일반적인 불가능성으로 해석하지 않지만 지금 확대 배포하지 않음. |
| B2. 자유형 argument 생성 head / 작은 decoder | 작은 생성기가 새로운 query나 필드 값을 충분히 잘 만들 수 있는가? | 고정 4-way head와 다른 문제. 새 문자열을 만들려면 decoder·vocab·copy 구조와 훨씬 많은 supervision 필요. query 의미 오류가 모든 후속 retrieval을 바꿈. | 현재 24질문에는 과도함. frozen LLM의 argument generation 유지. |
| B3. hidden→retrieval vector | 텍스트 query 없이 검색 가능한가? | retriever 표현·index 계약까지 바뀜. 외부 keyword API 인자를 대체하지 못함. LAnR와 매우 가까움. | 현재 연구범위에서 보류. |
| C. 품질을 유지하는 interaction 압축 | 어떤 관측/생성/도구 단계는 생략·대체·결합할 수 있고, 언제 새 생성에 투자해야 후속 단계가 줄어드는가? | 환경 관측의 인과 의존성, 안전한 단축 target 부족, 조기 종료 편향, 실제 비용 증가 가능성. | 연구질문은 가장 흥미롭지만 head 학습보다 압축 가능한 여지의 oracle 진단이 먼저. |

**실용 목적이면 A만으로도 충분히 가치가 있을 수 있다.** 여러 환경에서 품질을 유지하고 실제 처리비용을 줄인다면 도구 분류기가 단순해도 유용하다. 논문 목적이면 “분류기가 도구를 잘 맞힌다”만으로 강한 기여가 되기는 어렵다. 적은 데이터에서의 일반화, 어떤 실패에서 무너지는지, 동일 계산에서의 정책 이득, 잘 정의된 압축 가능성 또는 음성 결과가 필요하다. 현재는 독창성을 주장하지 않는다.

A의 3-hop 전이는 **같은 도구 집합에서 horizon 변화**다. 다른 agent·다른 tool schema·미지 도구 일반화와 같지 않다. 고정 4-class 출력 head가 새로운 도구 이름을 지원하는 것도 아니다. 100문항을 추가하면 추정은 나아지지만 무엇을 측정하는지의 혼동은 해결되지 않는다.

## 3. diffusion step distillation 비유의 유용한 부분과 한계

[Progressive Distillation](https://arxiv.org/abs/2202.00512)은 deterministic diffusion sampler의 여러 계산 단계를 더 적은 단계의 모델로 모사한다. 이 발상은 “좋은 teacher의 결과를 더 적은 순차 계산으로 재현한다”는 목표에 유용하다.

다만 RAG에서 `search → 새 entity 관측 → 다음 search`는 계산 반복만이 아니다. 다음 검색어를 결정할 정보가 실제 첫 검색 결과에서 처음 생길 수 있다. 해당 관측을 얻지 않고 같은 결정을 해야 한다면 head는 그것을 보장할 수 없다. 학습 때 teacher의 미래 결과·정답으로 target을 만드는 것은 허용되지만, inference feature나 호출 인자에 미래 ID·entity·gold support를 넣는 것은 누출이다.

압축은 세 가지를 따로 세야 한다.

1. **primitive tool-call 감소:** 중복 read를 생략하거나 한 번의 좋은 query가 두 번의 검색을 대신한다. 더 넓은 top-k로 같은 정보를 얻었다면 문서 수·입력 토큰·검색비용도 센다.
2. **LLM decision/prefill turn 감소:** search 뒤 관측된 결과를 정해진 규칙으로 read하는 실행 option은 도구 2회를 그대로 수행하면서 LLM 결정을 1회 줄일 수 있다. 이것은 도구 호출 감소가 아니다. 관측 후 실행 규칙은 허용되지만 실행 전부터 미래 ID를 아는 것으로 구현하면 안 된다.
3. **autoregressive decode 단계 감소:** tool 이름·구문을 head/일괄 prefix로 대체하거나 기존 인자를 선택한다. 현재 v3의 확인된 작동은 주로 이 범주다.

diffusion 비유는 2·3의 순차 계산 압축에 더 가깝다. 1에는 정보 획득의 하한이 개입한다. 불필요한 suffix를 없애는 early stop, 중간 단계를 대체하는 shortcut, 여러 primitive를 수행하는 option도 서로 구별해야 한다. 지금 FINAL만 더 빨리 선택하게 만들고 “trajectory distillation”이라고 부르면 실제 기여를 부풀리게 된다.

첫 진단은 현재 4-action 계약 안에서 수행한다. 새 macro를 넣는 것은 후속 별도 프로필의 schema/parser/handler/학습 target 변경이므로 이번 제안의 최소 실험에 포함하지 않는다.

## 4. 선행연구와 차별화 가능성

아래는 이번 검토에서 arXiv 1차 페이지를 다시 확인한 결과이며, 저자 실험을 재현한 것이 아니다. 전체 novelty search를 완료했다는 뜻도 아니다.

| 선행연구 | 이미 겹치는 주장 | 남겨 볼 질문 |
|---|---|---|
| [LAnR](https://arxiv.org/abs/2604.17866v2), [방법 본문](https://arxiv.org/html/2604.17866v2) | 같은 hidden state에서 latent retrieval와 경량 stop head; 텍스트 질의/종료 생성 축소 | 외부 API·기존 index를 유지하는 선택/생성 계약, 동일 backbone 비용에서의 policy 효과. 단순 hidden head는 차별점이 아님. |
| [GRASP](https://arxiv.org/abs/2607.10463v1) | keyword/semantic/paragraph reading, 답변·근거·turn 효율 학습 | frozen backbone의 적은 데이터 readout과 정확한 비용 분해. 같은 3 tools나 turn reduction 자체는 기여가 아님. |
| [Stop-RAG](https://arxiv.org/abs/2510.14337) | trajectory 기반 미래 가치로 stop/continue 제어 | argument의 선택-vs-생성 투자가 미래 interaction을 바꾸는 조건. stop/continue 가치라는 표현만 바꿔서는 충분하지 않음. |
| [AnySearch](https://arxiv.org/abs/2609.00813) | budget-aware search 정책과 정확도·효율 최적화 | 동일 비용 예산에서 작은 head의 제한과 효과. 예산 최적화 자체는 이미 선행. |
| [Partial Answer Quality and Utility](https://arxiv.org/abs/2609.16453) | 중간 답변 품질과 추가 iteration의 효용을 분리해 예측 | 현재 confidence 대신 action/argument별 실제 반사실 차이를 사용하는지. 강제 중간 답변을 매번 생성하면 그 비용도 포함해야 함. |
| [TabAgent](https://arxiv.org/abs/2602.16429) | 생성형 closed-set 결정을 trace 기반 classifier로 대체 | 같은 prefill 공유와 RAG 상태의 어려움에 대한 검증. classifier 교체 자체는 새로운 원리가 아님. |

가장 검토할 만한 연구 가설은 **“현재 관측에서 어떤 인자를 선택하거나 새로 생성할 때, 답변 품질과 남은 순차 interaction 비용이 어떻게 달라지는가를 학습한다”**이다. 예를 들어 더 긴 query 생성이 다음 search/read를 줄일 수도 있고, 짧은 read ID 선택이 중요한 증거를 놓쳐 전체 비용을 늘릴 수도 있다. 이를 현재 teacher exact-match CE는 학습하지 않는다. 실제 반사실 차이가 존재하고 단순 heuristic보다 예측 가능해야 비로소 후속 연구 후보가 된다.

## 5. learned policy 이득과 실행 구현 이득을 분리하는 방법

한 번의 prefill 뒤 head가 결정한다고 4B backbone 계산이 사라지는 것은 아니다. 현재 구조는 같은 backbone prefill을 수행하고 tool decision decode를 줄인다. 총비용은 대략 `prefill + action/syntax + argument decode + answer decode + query embedding/tool + runtime`이다. prefix batch와 decode 1토큰은 수행 방식이 달라 단순 token 합은 FLOPs도 latency도 아니다.

다음 비교를 혼동하지 않아야 한다.

| 비교 | 분리하는 효과 | 주의 |
|---|---|---|
| 같은 저장 상태·같은 action에서 native 생성 vs aligned cached argument 생성 | cache·syntax·grammar 실행 경로 | oracle action을 공짜로 준 상한 진단이며 실제 controller 속도가 아님. 이미 24상태 기능 진단이 있지만 비용/경로 일반성의 완전 증명은 아님. |
| 같은 상태·같은 실행기에서 native action selector vs learned head | 결정 방식의 차이 | native selector에 필요한 token/forward를 모두 포함. 고정 상태 결과와 자율 rollout 결과를 분리. |
| turn-only/직전결과 empty 여부 등의 작은 baseline vs hidden head | 단순 상태 shortcut 대비 정보 가치 | baseline도 train24에서만 정하고 validation은 이미 노출된 진단 자료임을 명시. |
| 동일 primary에서 read generate vs pointer vs 선택/생성 gate | argument 제어의 차이 | 후보·관측·checkpoint를 맞추고 검색인자 생성은 고정. 미래 경로 차이는 실제 rollout으로 확인. |

**native-first-token baseline에는 형식상의 함정이 있다.** 현재 native 첫 출력 token의 `<tool_call>`은 세 검색/읽기 도구가 공유하므로 그대로 4-way 예측값이 되지 않는다. exact native serialization의 첫 구별 가능한 action-name 분기까지 생성/점수를 계산하고 그 비용을 포함하거나, A/B/C/D 단일 token 선택 prompt를 명시적으로 따로 만들어야 한다. 후자는 같은 관측을 보더라도 prompt가 바뀐 baseline이다. 기존 head와 “동일 입력 token”이라 부르면 안 된다. 출력 token label이 여러 토큰이면 첫 토큰만 비교하는 것도 부정확하다. 기존 first-token 실험의 다른 prompt·generator 결과를 새 통제 비교인 것처럼 재사용하지 않는다.

fallback은 공짜가 아니다. head가 낮은 confidence이면 native decode를 **같은 보존 cache에서** 이어갈 수 있는지 확인하고, 중복 prefill이나 이미 만든 payload가 있다면 전부 센다. validation8로 정교한 calibration curve를 맞추기는 어렵다. probability threshold sweep을 많이 해 best를 고르는 것보다 미리 정한 소수 threshold와 coverage-risk를 보고하고, 근거가 부족하면 always-generate/native 기준선을 유지한다.

## 6. 지금 할 최소 실험: 24+8 질문에서 압축 가능한 여지와 실행 비용 분리

**첫 행동은 새 head 학습이나 100/30 평가가 아니라, 기존 train24의 각 질문에서 한 상태를 고정한 제한된 반사실 continuation 진단이다.** 방금 CPU audit은 준비 근거이며 아래 GPU continuation은 아직 실행하지 않았다. 이 실험은 C를 추진할 가치가 있는지를 먼저 묻고, A의 실행 이득을 검증할 작은 고정상태 비교를 함께 계측한다. 새 정책 배포나 RL이 아니다.

### 샘플·누출 계약

- 기존 train24/validation8/oldtest20 split은 변경하지 않는다. oldtest20과 새 100/30 질문은 이번 진단에 쓰지 않는다.
- train24의 teacher trajectory에서 **첫 tool 결과를 받은 직후** 상태를 질문마다 정확히 1개 고른다. 원 teacher는 tool2~3이므로 압축/대체할 다음 단계가 있다. 상태 선택은 정답 여부·최종 품질을 보지 않는다. 24개 prefix를 먼저 hash로 동결한다.
- 이미 준비한 train24의 첫 8질문은 먼저 기능 smoke에 사용할 수 있다. 그 질문의 준비된 여러 상태 중 위 규칙에 맞는 1개씩만 택한다. smoke는 cache 독립성·누출·parser·계측·동일상태 재현을 검증하며 성능에 따라 문항을 교체하거나 후속24개를 선택하지 않는다. 공통 준비 기록은 `lab/registry/text_integrated_20261016/head_direction_common_readiness_20261017.json`이다.
- validation8도 같은 규칙으로 1상태씩 정하되 train 진단 후 branch 규칙·비용 계수·gate를 고정한 뒤 실행한다. 이는 이미 노출된 validation의 확인 진단이며 untouched test가 아니다. train/validation을 섞어 학습하거나 성능을 합쳐 heldout 주장하지 않는다.
- 질문·그때까지 실제 받은 관측·기존 도구 schema만 정책 입력이다. gold는 branch 결과 채점과 oracle target 계산에만 쓴다. prefix 뒤 teacher 인자·결과·ID를 숨긴다. 동일 초기 cache를 branch별 독립 복제/재계산하고 cache mutation 교차 오염을 막는다.

### 동일 상태의 branch

각 상태에서 최대 5 continuation을 수행한다. (1) frozen action head+generated args의 정상 continuation, (2) native selector+동일 aligned 실행기의 continuation, (3) 즉시 FINAL, (4~5) 정상 head와 다른 합법 tool action들이다. 다른 tool은 최대 2개이며 read는 실제 관측 ID가 있을 때만 합법이다. 초기 선택 뒤 인자는 같은 frozen Qwen이 현재 상태에서 새로 생성하며, 후속은 모두 frozen primary로 계속한다. native 조건도 최초 selector 차이를 격리하는 branch이며 native 전체 정책 성능과 혼동하지 않는다. native가 FINAL이면 답을 반환하고 그 비용을 모두 센다.

초기 head action이 FINAL인 상태에서는 세 tool을 전부 시험하면 5개를 넘을 수 있다. 이 경우 keyword→semantic→read의 사전고정 순서로 최대 2개만 시험하고 빠진 action을 기록한다. 최소 실험은 포괄적인 action oracle이 아니라 **시험한 branch 집합의 oracle 상한**이다. 모든 실제 tool은 현재 계약 안에서 실행하며, teacher suffix를 그대로 붙여 “잘 단축됐다”고 채점하지 않는다.

native selector branch는 아직 구현·검증된 기능이 아니다. native action-name이 확정될 때까지 필요한 구문 생성 비용을 포함하고, 가능한 경우 그 cache를 이어서 인자를 생성한다. 이미 생성한 prefix를 버리고 다시 처리하거나 추가 prefill이 생기면 그 비용도 기록한다. 불필요한 중복을 가진 native 구현과만 비교해 learned head 우위를 선언하지 않는다. 이런 정렬이 기능 smoke에서 성립하지 않으면 우선 기존 native 전체 경로를 별도 시스템 baseline으로 남기고, selector만의 인과 비교는 미완료로 표시한다.

추가로 train에서 hash순 8상태만 native action을 고정해 cached 경로와 native 경로를 replay한다. 목적은 실행 경로/결과 parity와 비용 분해다. 이 공짜 action 비교를 deployable baseline이나 속도 주장으로 사용하지 않는다.

### 비용·예산·baseline

- 최대 32상태×5 = **160 continuation**, 고정상태 추가 비교 최대 16건. 새 학습 없음, seed·모델·prompt·retriever·head 모두 동결.
- continuation당 남은 primitive tool 최대 2회와 FINAL, 출력 상한은 기존 512tokens/call 및 context 16,384 유지. 예산 소진은 모든 branch에 동일한 명시적 FINAL을 적용하고 자연 종료와 구분한다. 이 제한은 원 horizon 전체 평가가 아닌 짧은 반사실 probe의 계약이다.
- 전체 hard cap 제안: 모델 로딩 제외 **30분, generated tokens 120,000, 누적 입력+forced-prefix tokens 2,000,000** 중 먼저 도달하는 값. 정해진 질문 순서에서 멈추고 중단/미완료로 기록한다. 품질이 좋거나 나빠서 중간 종료하지 않는다. train phase만 완료해도 별도 결과를 남기며 부족분을 임의로 추가 수집하지 않는다.
- branch들에 같은 상한을 주는 것만으로 realized cost가 같아지지 않는다. 비용별 Pareto를 보고, 고정 remaining-tool budget B=0/1/2의 native/primary 기준점과 비교한다. B=0은 FINAL이고 B=1은 해당 trace의 **첫 실제 tool 결과 이후 별도 answer probe**가 필요하다. 저장된 최종 답을 당겨 쓰지 않는다. B=1 probe는 32상태×2 baseline 최대64건을 별도 상한 내에서 수행하여 총 최대240건으로 제한한다.
- 품질 비교는 EM/F1·근거 recall·관측된 ID citation 유효성·format error를 함께 본다. 시간은 prefix 구축 비용과 suffix 비용을 분리하고 실제 전체 trajectory 비용도 보존한다. full-prefill 수/길이, batch prefix 길이, decode forward, generated/forced tokens, query embedding과 tool walltime을 모두 기록한다. 공유 GPU에서는 latency를 관측값으로 보고한다.

### oracle 목표와 사전 중단 기준

같은 초기 상태에서 branch의 결과 `(QA, evidence, calls, decision turns, compute)`를 모아, **품질을 지키는 시험 branch 중 비용이 가장 낮은 것**을 oracle로 정의한다. 최종 gold를 쓰는 특권적 진단이며 학습된 정책 성능이 아니다. 결과가 모두 F1=0인 질문에서 빨리 틀린 것을 긍정적 압축 사례로 세지 않는다. 정상 continuation이 맞은 문항의 정답·근거 보존과, 틀린 문항의 실제 품질 개선을 별도로 보고한다.

후속 C 학습으로 넘어갈 탐색 gate를 다음처럼 제안한다.

- 정상 continuation이 EM=1인 질문이 train24에서 최소 5개 있어야 보존 여지를 평가한다. 부족하면 결론은 “자료 부족”이며 현 2-hop 집합으로 C 학습을 시작하지 않는다.
- 그 정답 문항들에서 oracle가 EM을 모두 보존하고 질문별 F1·관측 근거 recall을 낮추지 않으면서, 총 decision turns 또는 primitive tool calls를 **20% 이상** 줄여야 한다. 해당 성공이 최소 3개의 독립 질문에서 나와야 한다. 품질 보존 가능성으로 branch를 고른 oracle라서 성공은 학습 가능성 증명이 아니며, 실패는 시험 branch 집합의 여지가 작다는 뜻이다.
- 생성/계산 절약은 입력 증가를 포함한 비용 성분으로 함께 확인한다. model forward 분석에서 절감 방향이 불명확하거나 오직 자유롭게 늘린 top-k/읽기량으로 얻은 효과라면 효율 gate는 보류한다. 시간 우위는 공유 GPU 값만으로 판정하지 않는다.
- **FINAL만으로 모든 이득이 설명되면** early-stopping 방향으로 명명하고 Stop-RAG/utility baseline을 우선한다. nonterminal 대체가 최소 3질문에서 더 짧은 성공 suffix를 만들지 못하면 “중간 trajectory 압축” 학습은 시작하지 않는다.
- validation8은 방향과 실패 형태를 확인하는 자료다. 8문항 무손실을 noninferiority 증거로 선언하지 않는다. gate를 거기서 다시 조정하거나 oldtest20을 재튜닝에 쓰지 않는다.

20%·5질문·3질문은 투자 결정을 위한 작은 파일럿 기준이다. 논문 수락 기준이나 통계적 보장은 아니다. 특히 평균2~3 tool이라는 짧은 환경에서는 이 gate를 넘지 못할 가능성이 상당하며, 그 결과도 다음 방향을 정하는 유용한 정보다.

## 7. 하나의 로드맵과 대안 결정표

1. **지금:** 위 제한 진단의 상태·branch·계측 계약을 고정한다. 기존 head와 always-generated args를 유지한다. 이번 리뷰에서는 실행하지 않았다.
2. **압축 여지가 확인되면:** train 질문에서 얻은 반사실 return/cost 차이를 target으로 작은 가치 head 하나만 학습한다. 현재 행동 모방 head보다 나은지, FINAL-only·turn-only·native selector보다 나은지부터 본다. 처음부터 RL, 새 argument decoder, macro 정책을 함께 넣지 않는다. 학습 질문의 branch 결과를 validation feature에 흘리지 않는다.
3. **argument 선택에 따른 차이가 주요 원인이면:** 정적 4-slot 대신 현재 후보의 내용/ID span을 사용하는 pointer 또는 후보별 scorer와 GENERATE 선택을 검토한다. 동일 action을 고정한 비교로 시작한다. 후보 매칭 CE와 downstream cost-to-go supervision을 분리하고 검색어 생성은 frozen LLM으로 유지한다.
4. **정책과 gate를 동결한 뒤:** 미노출2-hop100/3-hop30 또는 variance 기반으로 정한 확인 규모로 paired 평가한다. 모든 노출 qid와 공유 decomposition family를 제외하고 배제 결과가 target population을 어떻게 바꾸는지도 기록한다. 100/30은 final/generalization 후보이지 보편적으로 충분한 표본수는 아니다. 3-hop30은 탐색적 horizon shift이며 별도 agent 전이로 확대 해석하지 않는다.

| 진단 결과 | 선택할 방향 | 보류할 것 |
|---|---|---|
| cached 실행만 절약, learned selector가 native보다 품질/비용상 별도 우위 없음 | A를 실용적 실행 최적화/정책 근사로 정리. 단순 baseline과 확인평가 수행 | learned policy 자체가 효율 원인이라는 주장 |
| head가 단순 순서 baseline보다 낫고 분포 변화에서도 저하가 작음 | A 일반화 연구. 실패복구 상태의 coverage를 필요한 만큼 추가 | 근거 없는 MLP 대형화·수많은 calibration sweep |
| oracle 이득 대부분 즉시 FINAL | C의 early stopping 하위 문제. Stop-RAG/utility와 비교 | diffusion-like 중간 transition 증류 주장 |
| nonterminal shortcut이 품질을 유지하고 실제 비용 감소 | C를 주 방향으로, 제한된 counterfactual value head 학습 | 처음부터 대규모 RL/새 tool macro |
| primitive calls는 그대로이고 LLM turns만 줄일 가능성 | deterministic/reactive option의 시스템 실험으로 별도 계약 설계 | tool-call reduction이라는 명명 |
| read/query 선택 차이가 이후 단계·품질을 크게 좌우 | B1을 C의 argument-value 문제로 좁혀 연구 | 자유형 argument 생성head·retriever 전체 교체 |
| oracle조차 절약 여지가 작거나 정답 사례가 너무 적음 | A 확인평가를 하거나 task/horizon 선택을 다시 검토 | 같은 2-hop 자료에서 압축 학습을 더 오래 돌리는 것 |

연구 우선순위를 정하는 데 사용자의 추가 승인이 반드시 필요한 운영 질문은 없다. 질문 수·hash 선택·계측·고정 baseline은 위 범위에서 구체화할 수 있다. 다만 “현재 시스템의 실용 최적화”와 “더 강한 논문 기여를 위해 task/horizon까지 바꾸기”는 목표가 다른 선택이다. 현 검토는 후자의 가능성을 작은 진단으로 먼저 확인하되, 성과가 없으면 A의 실용 가치를 보존하는 경로를 추천한다.

## 실행 상태와 provenance

- 새 학습, GPU 실험, 새 테스트셋 평가, RL, step distillation: **미실행**.
- 검토 대상 초안: `docs/JEV_HEAD_GENERALIZATION_DRAFT_20261017.md`.
- 미검증 prototype: `/content/local_fast/lsdrag-text-v1/drafts/text_head_generalization_20261017.py`로 격리되었으며 이 검토에서 실행하지 않았다.
- frozen head SHA256: `534da7b64150878869850cf7ccb735ec0fa8fbd643ddd0a74db19afeea4ebaef`.
- 기계 판독 요약: `lab/reports/text_integrated_20261016/astra_head_direction_review_20261017.json`.
- Git/Drive 저장과 memory 갱신은 부모 작업에서 수행한다. 본 검토는 기존 runtime 코드를 수정하지 않았다.
