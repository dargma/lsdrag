# 단일 LLM의 decision policy head — 연구 가설과 단계별 실험 초안

2026-10-23. 사용자 논의 정리 및 계획 저장. 구현/학습/새 평가 미실행. 기존 protocol과 checkpoint는 역사적 기준선으로 보존한다. 새 실행 protocol은 아직 동결하지 않았다.

## 목표와 기여 가설

별도 decision backbone 없이 동결 Qwen3-4B-Instruct-2507의 동일 prefill 표현을 작은 head가 읽어 keyword_search / semantic_search / read_chunk / FINAL을 선택한다. 같은 turn의 KV cache로 필요한 인자와 답변을 생성한다. 학습은 teacher 행동 모방에서 출발하되, 전체 trajectory 결과와 남은 비용으로 정책을 개선한다.

두 효과를 분리해 검증한다: (1) 도구명/고정 구문을 순차 생성하지 않아 발생하는 생성 비용 절감, (2) 정책이 불필요한 검색·읽기·실패 반복을 줄여 발생하는 trajectory 비용 절감. 현재는 (2)가 입증되지 않았다. 단순 head 부착/도구 dispatch 자체의 novelty를 주장하지 않는다. 구조·학습·동일 품질에서의 실제 시스템 비용을 함께 입증하는 것이 후보 contribution이다. 실패 후 추가 탐색이 필요한 경우 step 증가는 허용한다.

## 현재 증거

src/lsdrag/text_integrated.py capture는 output.last_hidden_state[0,-1]을 사용한다. action linear와 read-slot linear는 독립적이다. 기존 teacher train24/validation8은 작은 진단용 분할이며, teacher 정답 여부와 무관한 행동모방 데이터다. Frozen100은 이미 평가에 노출됐으며 후속 학습·튜닝에 사용하지 않는다. 기존 결과는 learned sequential policy improvement 증거가 아니다.

## 공개 decision 모델 조사 (2026-10-23 확인)

- D3-Flash: Qwen3.5 계열 backbone과 answer-code readout. 마지막 위치에서 선택지 점수를 계산하며 질문별 forward를 수행한다. vLLM 공식 문서는 full-attention layer의 bidirectional 처리를 명시한다. 현재 causal Qwen head와 attention/KV 조건이 다르므로 그대로 이식할 수 없다. [공식 recipe](https://recipes.vllm.ai/vllm-sr/d3-flash), [model card](https://huggingface.co/vllm-sr/d3-flash). 공개 코드 13dc1cc8ef23caefa765907e1d764751e5629f7d의 modeling_d3.py/decision_config.json도 확인: 255-way answer-code readout, FP32 readout, temperature metadata. 코드는 다운로드/정적 확인만 했다.
- D3-Turbo: state를 요청당 한 번 읽어 여러 질문에 답하도록 개선. [card](https://huggingface.co/vllm-sr/d3-turbo). 조사 revision a0f3e9e59ebd1202fe1d4de1c4c0f9a986758cbb. 우리 목표는 이 외부 모델을 추가하는 것이 아니다.
- D3의 공개 card/runtime에서 상세 학습 데이터 manifest, mixture, 전체 loss/trajectory RL recipe는 확인하지 못했다. 출력 확률이 있다고 downstream 성공 확률을 학습했거나 RL 모델이라고 단정할 수 없다. 공개 평가 suite를 학습 데이터로 오인하지 않는다.
- 학습 데이터가 명시된 별도 사례인 Decision-1.0-Route: 10개 task, 366,323 distinct rows; domain/tool need/safety/PII/hallucination 등의 라벨 데이터. Toucan, glaive function calling, RAGBench 등이 포함된다. 이것을 D3 학습 데이터로 대입하지 않는다. [훈련 기록](https://huggingface.co/vllm-sr/Decision-1.0-Route-0.6B/blob/main/TRAINING_DATA.md).
- Vela 2.0는 causal token 표현과 label pooling 설계가 성능에 영향을 주는 사례를 공개한다. 이는 span task의 관찰이며 우리 행동정책으로의 이득은 미검증이다. [공식 기술 설명](https://vllm-sr.ai/blog/vela-2-0-open-foundation-routing-models/).
- FORGE는 frozen host 주변의 작은 router를 outcome/cost enumeration, KL warm-start, GRPO로 학습한다. 이 선행 때문에 작은 router+RL 조합 자체를 새롭다고 할 수 없다. 우리 가설은 동일 prefill 내부 readout과 매 단계 head가 유지되는 sequential policy의 결합이다. [저자 페이지](https://xixiaouab.github.io/projects/FORGE/).
- 비용을 고려한 정책 반복의 관련 토대: [Ross & Bagnell 2014](https://arxiv.org/abs/1406.5979). On-policy actor-critic 후보: [PPO](https://arxiv.org/abs/1707.06347). 아래 설계는 우리 제안이며 이 논문들이 본 실험의 성공을 보장하지 않는다.

## 입력과 구조

최초 추천은 작은 residual MLP actor-critic, 구조 확장은 행동별 attention pooling이다. LLM은 동결하며 backbone 추가 forward 없이 이미 계산한 hidden states를 사용한다. 전체 token을 읽으면 메모리/읽기 비용이 늘므로 별도 계측한다.

| 축 | 비교 후보 | 해석/통제 |
|---|---|---|
| 토큰 위치 | 마지막 토큰, 질문 끝, 최근 관측 끝 | causal 질문 위치는 이후 관측을 못 본다. 정보 가용성 차이까지 명시 |
| 집약 | 마지막 k 토큰 평균, 관측 span 평균, 마지막 토큰+관측 집약 | 차원을 맞춰 위치 효과와 파라미터 증가를 구분 |
| 구조 | linear, 작은 residual MLP, 행동별 query attention pooling | MLP를 필수 baseline으로 두고 추가 파라미터/메모리/시간 보고 |
| 후순위 | 여러 layer의 학습형 혼합, 작은 recurrent memory | 작은 자료에서 과도한 동시 탐색을 피함; 앞 단계 실패 원인이 있을 때만 |

행동별 attention 후보: H → 공유 저차원 K/V projection → 네 행동 query가 각기 evidence 집약 → shared scorer → masked policy logits. 별도 작은 value/성공·비용 출력 branch. 이를 한 policy-head 모듈로 구현할 수 있지만 출력 branch가 하나뿐이라고 주장하지 않는다. learned query는 고정 4행동용이며 새 tool zero-shot 일반화는 별도 과제다. attention weight는 인과적 설명으로 간주하지 않는다.

## 출력과 확률의 역할

출력 계약 초안: selected_action, action_probs[4], success_probs[4] (학습·교정된 경우만), expected_remaining_cost[4], legal_mask, policy_version. 현재는 제안으로 기존 head가 success_probs를 지원하지 않는다.

- π(a|s): 행동 선택 분포, 합은 1. 모방학습에서 높은 값은 teacher 일치 확신일 수 있으며 성공 보장이 아니다.
- p(success|s,a,πk): 해당 행동 이후 version πk로 계속했을 때의 terminal success 추정. 행동간 합 1 조건 없음. 성공 정의(EM 등)를 고정해야 한다. F1 회귀값은 성공 확률이 아니다.
- Q(s,a): 성공/품질과 비용을 결합한 기대 return. 성공 확률과 동일하지 않다.
- Value/critic은 학습 때만 사용해 추론 비용을 줄일 수 있다. 실행시 성공·비용 확률도 제공하려면 해당 branch 비용을 포함한다.
- 활용: on-policy stochastic exploration, 불확실/행동간 가치차이가 작은 상태의 추가 수집, 신뢰도 교정 평가, 비용 선호 변화에 따른 policy 선택. entropy가 epistemic uncertainty 자체는 아니다.
- 후보를 확률이 낮다는 이유로 hard-prune하면 개선 행동을 영구히 놓칠 수 있다. 합법성 mask와 exploration floor, 별도 branch budget을 둔다. top-k 실제 rollout은 추가 추론비용이므로 기본 추론 경로가 아니다.

## 단계별 실험 초안

### 0. 재현과 비용 분리
기존 native와 BC head를 보존. optimized native-name/constrained-format 생성 baseline을 구현/검증해 syntax 절약과 정책 효과를 분리한다. 모델/인자생성기/도구/프롬프트/실행 cap 고정. 기존 raw와 runtime 환경부터 검증한다. 복원만으로 실행 준비 완료라고 하지 않는다.

### 1. 작은 표현·구조 진단
기존 train24/val8에서 마지막 위치 linear/MLP, 관측 pooling+MLP, 행동별 attention을 소규모 비교. 동일 teacher label 비교는 표현 진단일 뿐 정책 개선 증거가 아니다. 기존 NPZ는 마지막 feature만 있으므로 다른 위치는 고정 입력을 재prefill하여 추출해야 한다. 이 진단으로 최종 QA 설정을 평가100에 맞추지 않는다.

### 2. outcome 기반 데이터
현재 πk가 방문한 train 상태에서 합법적 대안 행동을 실제 실행하고, 이후 모든 행동도 동결 πk로 끝까지 실행한다. Native로 복귀하지 않는다. LLM 인자·답변 생성 설정은 고정한다. state/행동/실제 인자/관측/behavior logprob/정책 version/terminal outcome/남은 비용을 저장한다. 미래 outcome은 label에만 사용하며 정책 입력으로 누출하지 않는다. 모든 성공과 실패를 보존한다.

제안 pilot: 기존 train24의 질문당 최대2상태×합법 행동 최대4×2 suffix samples = 최대384 continuation. 실제 상태수가 적으면 줄인다. 이는 진단 상한이며 기존 budget에 자동 승인된 실행량이 아니다. 기존24문항만으로 일반화 결론을 내리지 않는다. 다음 확대는 미노출 training questions를 기존 분할과 family 중복 제거하여 새 manifest로 추가하고, 기존 test는 그대로 보호한다.

한 번의 deterministic branch 결과는 성공확률 추정이 아니다. 정책 sampling/반복 또는 다른 상태들에 대한 supervised generalization이 필요하다. 정책 갱신시 성공 확률의 continuation 조건도 바뀌므로 재수집/교정한다.

### 3. head-only trajectory RL
BC 초기화 actor + 작은 critic. 매 iteration에서 현재 head가 모든 step을 선택한 on-policy trajectories를 수집하고 PPO-style update로 actor/critic만 갱신. Terminal 품질과 step별 비용으로 return/advantage 계산. 기존 BC와의 KL anchor 및 entropy는 collapse 방지 후보이며 validation으로 제한적으로 선택한다. PPO old logprob은 실제 수집 정책과 일치해야 한다. 임의 branch/off-policy 데이터를 on-policy PPO trajectory인 것처럼 합치지 않는다.

반사실 분기 데이터는 critic pretraining/성공·비용 auxiliary supervision 및 outcome-based CE baseline에 사용한다. 모든 action에 같은 episode return을 무차별 복제하는 것보다 state별 cost-to-go와 advantage를 사용한다. learned p(success) 자체를 유일한 보상으로 쓰지 않고 실제 terminal score로 검증한다.

필수 비교: BC + 고정 구조, outcome CE + 같은 구조, trajectory RL + 같은 구조. 이후 가장 유망한 두 구조만 같은 RL/rollout budget으로 비교. RL 이득과 구조 이득을 분리하며 가능한 3 seeds, 비용이 부족하면 1-seed 탐색임을 명시한다.

### 4. 정확도·시간 공동 최적화
초기 정확도 우선, 이후 J = E[quality - λ·remaining_cost]. quality는 사전 정의한 EM/F1/근거 지표로 구분. 실행 정책에는 현재 관측과 남은 budget만 제공. 품질 하한을 만족하는 정책 중 비용이 낮은 것을 선택하는 constrained formulation도 비교한다.

공유 GPU wall time은 잡음이 크다. pilot은 실제 prefill/forced-prefix/decode tokens, 도구 calls 등 비용 성분을 따로 기록하고 고정 proxy를 사용한다. 이후 통제된 반복 시간 측정으로 전체 episode latency, p50/p95, timeout/failure 포함을 검증한다. 출력 토큰 감소만으로 시간 개선을 주장하지 않는다. λ를 test에서 고르지 않는다.

## 평가와 통과 기준

- 최종 EM/F1/support, 실패 후 recovery, premature FINAL, tool calls, 불필요 반복, 전체 token/시간/수집비용을 보고.
- 확률: terminal-success Brier/NLL 및 reliability plots; policy entropy와 분리. validation에서 교정, test에서 평가.
- 같은 질문에서 BC/outcome CE/RL을 paired 비교하고 질문 단위 uncertainty 보고. trajectory 중 state를 독립 표본처럼 세지 않는다.
- 새 미노출 평가 manifest 동결; exposed100은 회귀 참고만, 튜닝/새 holdout 주장 금지.
- contribution 통과: BC뿐 아니라 같은 label의 CE와 optimized native 대비 quality-cost 이점; 정책이 실제 후속 행동을 개선했다는 실패/회복 증거. 우월성은 데이터가 뒷받침할 때만 주장.
- 순차 탐색을 줄이되 필요할 때 더 탐색하는 능력을 보존. 최소 step 자체를 성공 지표로 삼지 않음.

## 복원 및 실행 상태

local-fast에 Qwen 13파일, embeddinggemma-2 15파일, MuSiQue train/dev를 pinned public source에서 받아 SHA 검증. unique teacher32 JSON/NPZ와 v3 head archive를 Drive에서 복원/검증. head SHA 534da7b64150878869850cf7ccb735ec0fa8fbd643ddd0a74db19afeea4ebaef. provenance: lab/registry/recovery_20261023/recovery.json. GPU 로드/새 학습/실행환경 smoke는 하지 않았다. 데이터·모델 복원 완료와 runtime 준비 완료를 구별한다.

다음: 이 초안에서 작은 pilot의 reward, budget, split, 비교 조건을 실행 protocol로 확정하고 환경 smoke/구현 검토. 자동 실행 없음.
