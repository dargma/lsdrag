# Surprise와 world model: decision 품질 연구 가능성 검토

2026-10-15 UTC. 1차 논문·학회 원문을 검색하고 관련 방법/초록을 확인했다. 재현 결과가 아니며 아래 모델 설계와 실험은 가설이다. 현재 full A-RAG 5 tools+FINAL direct-dispatch 코드는 유지한다. 이 문서는 새 GPU 실행이나 구현을 승인·실행한 기록이 아니다.

## 판단부터

사용자의 '예상한 관측과 실제 관측의 차이를 이용해 다음 판단을 개선한다'는 방향은 world-model/curiosity 연구와 정확히 연결된다. 다만 **surprise를 보상하거나 task-relevant prediction error를 쓰는 발상 자체는 이미 강한 선행연구가 있다.** 특히 2026-09-26의 DRPE는 제목부터 decision-relevant error와 planning quality를 연결한다. 따라서 기존 access auxiliary 위에 world model·surprise·memory를 모두 추가하는 방향은 추천하지 않는다.

추천은 **별도의 작은 반증 pilot로 기존 access-auxiliary 가설을 대체할 가치가 있는지 비교**하는 것이다. 새 중심 가설은 '예측이 어긋난 관측 중 다음 행동의 순위를 바꾸어야 하는 정보만 구분해 state update에 반영하면, 같은 정보와 용량을 받는 generic Q보다 실패 후 회복과 최종 성공이 좋아지는가'다. 가장 큰 장점은 RAG 밖 agent에도 적용 가능한 문제 정의다. 신규성은 아직 확인되지 않았으며 positive result만으로 ICML 기여가 보장되지 않는다.

## 개념을 구분해야 한다

상태 `h_t`, 선택한 행동과 실제 인자 `(a_t,u_t)`, 다음 관측 `o_(t+1)`에 대해 latent predictor가 `z_hat = F(h_t,a_t,u_t)`를 계산하고 실제 관측을 `z = E(o_(t+1))`로 encoding한다. `d(z,z_hat)`은 prediction error다. density model이 있으면 `-log p(o|h,a,u)`는 surprisal이다. 두 값은 정의부터 같지 않다.

- **Prediction error:** 무지·모델 bias·표현의 부적절함·관측 노이즈가 모두 섞인다.
- **Epistemic uncertainty:** 데이터가 더 있으면 줄일 수 있는 model uncertainty다. 한 predictor의 residual만으로 분리되지 않는다.
- **Information gain:** 관측 전후 belief/model posterior의 변화다. 큰 likelihood error와 동치가 아니다.
- **Task utility:** 해당 관측 때문에 더 좋은 행동을 택해 얻는 최종 reward 차이다. 이미 아는 중요 사실은 surprise가 낮아도 utility가 높을 수 있고, 랜덤 문자열은 surprise가 높아도 utility가 없다.
- **Value prediction error / TD error:** reward와 bootstrap value의 차이로, observation prediction error와 다른 대상이다. residual을 Q update에 넣는 것이 새로운 RL 업데이트라고 주장하면 안 된다.

현재 읽은 이미지로 feature를 만든 뒤 '이미지를 읽기 전에 무엇을 볼지 골랐다'고 주장할 수 없다. 실제 `z`는 행동 이후에만 이용 가능하다. **현재 residual은 다음 iteration의 state update에 쓸 수 있고, 현재 action 선택에는 predicted distribution/지난 residual만 쓸 수 있다.** cheap retrieval embedding으로 actual image latent를 얻으면 VLM read 전 계산할 수 있지만 비용을 포함하고 OCR·속성의 표현 손실을 확인해야 한다.

## 기초 원리와 직접적인 신규성 위협

| 제목·최초 arXiv 날짜 / 1차 출처 | 무엇을 실제로 하는가 | 우리에게 주는 제약 |
|---|---|---|
| **VIME: Variational Information Maximizing Exploration**, 2016-05-31. [원문](https://arxiv.org/abs/1605.09674) | Bayesian dynamics model belief의 information gain으로 intrinsic exploration reward를 만든다. | surprise와 posterior uncertainty reduction 구분은 오래된 핵심 문제. |
| **Curiosity-driven Exploration by Self-supervised Prediction**, 2017-05-15. [ICM](https://arxiv.org/abs/1705.05363) | inverse-dynamics로 학습한 visual feature에서 action consequence의 forward prediction error를 curiosity로 사용한다. | '픽셀 대신 latent를 예측하고 residual로 exploration'은 신규 발명이 아니다. |
| **Exploration by Random Network Distillation**, 2018-10-30. [RND](https://arxiv.org/abs/1810.12894) | 고정 random observation encoder를 맞추는 predictor error를 exploration bonus로 사용한다. | action-conditioned dynamics 예측과 다르다. novelty-only 저비용 baseline으로 적합. |
| **Dream to Control: Learning Behaviors by Latent Imagination**, 2019-12-03 / ICLR 2020. [Dreamer](https://arxiv.org/abs/1912.01603) | latent world-model imagination에서 value gradient를 통해 policy를 학습한다. | world model의 용도는 surprise뿐이 아니다. 전체 Dreamer 학습을 우리 작은 head와 동등한 baseline으로 볼 수 없음. |
| **The Value Equivalence Principle for Model-Based Reinforcement Learning**, 2020-11. [NeurIPS 원문](https://proceedings.neurips.cc/paper/2020/file/3bb585ea00014b0e3ebe4c6dd165a358-Paper.pdf) | full dynamics 복원 대신 관련 policy/value에 대해 같은 prediction을 주는 model을 논한다. | '모든 관측을 정확히 예측할 필요 없고 decision에 필요한 것만'은 이미 확립된 원리. |
| **Value Gradient weighted Model-Based Reinforcement Learning**, 2022-04-04. [VaGraM](https://arxiv.org/abs/2204.01464) | observation error와 policy 목적의 mismatch를 value-gradient weighting으로 다루고 distractor state에 대한 robustness를 검토한다. | task-relevant error weighting이 일반적 신규성이라는 주장은 불가. |
| **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023-01-19. [I-JEPA](https://arxiv.org/abs/2301.08243); **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning**, 2025-06-11. [원문](https://arxiv.org/abs/2506.09985) | target latent prediction과 representation learning; V-JEPA2는 action-conditioned planning까지 연결한다. | **JEPA와 TypeSafe Jev는 다른 개념**. latent predictor 하나를 붙인다고 JEPA 재현이 아님. image/video 사전학습 전체를 새로 할 필요 없음. |
| **Calibrated Value-Aware Model Learning with Probabilistic Environment Models**, 2025-05-28, 최신 제목. [원문](https://arxiv.org/abs/2505.22772) | value-aware/MuZero 계열 surrogate의 calibration 한계와 수정, stochastic model과 auxiliary의 상호작용을 분석한다. | value-space loss가 항상 올바른 dynamics/value를 복원한다는 가정 금지. |
| **Can In-Context Learning Support Intrinsic Curiosity?**, **2026-06-17**. [원문](https://arxiv.org/abs/2606.19476) | ICL prediction errors와 counterfactual context로 learning progress reward를 구성하는 문제를 연구한다. 일반 MDP에서 unbiased 추정의 한계와 비시간적 subclass의 positive result를 구분한다. | 관측 오류를 곧바로 unbiased information gain이라고 부르지 않는다. frozen sequence model을 쓰더라도 이 문제가 사라지지 않음. |
| **Not All Errors Matter: Decision-Relevant Prediction Error Predicts Planning Quality**, **2026-09-26**. [본문](https://arxiv.org/html/2609.32322v1) | factored gridworld에서 decision-relevant dimensions의 error를 정의하고, **같은 total error에서 error allocation만 바꾸는 iso-error** 실험을 한다. task가 바뀌면 error relevance도 달라진다. | **가장 직접적인 개념 중복**. 'task-relevant surprise가 raw surprise보다 중요'만으로 부족. natural-language/multimodal 관측에서 relevance를 gold dimensions 없이 학습하고 실제 행동을 개선하는 추가 증거가 필요. |

## 최신 language-agent world-model 연구

| 확인한 논문·날짜 / 1차 출처 | 메커니즘과 현재 가설과의 관계 |
|---|---|
| **Current Agents Fail to Leverage World Model as Tool for Foresight**, **2026-01-07**. [논문](https://arxiv.org/abs/2601.03905) | world-model simulation을 제공해도 적절히 호출·해석·통합하지 못하는 agent 병목을 분석한다. world-model 추가가 곧 policy 개선이라는 추론은 부당하다. |
| **Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning**, **2026-02-10**. [AWM](https://arxiv.org/abs/2602.10090) | 코드·DB 기반 synthetic executable environments로 agent RL을 확장한다. language observation predictor나 surprise head와 동일 방법이 아니다. 비RAG 테스트 환경 후보 자원. |
| **Qwen-AgentWorld: Language World Models for General Agents**, **2026-06**, 2606.24597. [본문](https://arxiv.org/html/2606.24597v1) | 다양한 agent 환경의 response prediction과 LWM training을 연구하고, 실제 feedback과 예상 response의 일치 및 prediction-driven action refinement를 분석한다. | tool-response 예측+agent 성능 개선 자체는 기존 범주. full textual response를 생성하는 비용과 compact prediction의 정보 보존을 분리해야 한다. |
| **Agent-Editing World Model: Rethinking World Modeling for LLM Agents**, **2026-09-23**. [AEWM](https://arxiv.org/abs/2609.28416) | high-entropy tool response 복원의 한계를 지적하고 Action Judge(critical/exploratory/noisy)와 State Revision을 통해 task progress를 모델링하고 state를 수정한다. | **가장 가까운 language-agent 품질 위협**. '관측보다 task progress를 예측하고 나쁜 state를 수정'이라는 큰 줄기는 이미 있음. compact non-generative correction을 한다는 구현 차이만으로 충분치 않음. |

예상 tool observation을 실제 observation과 비교하는 정확한 선행연구 전수 조사가 끝난 것은 아니다. 위 경쟁 연구만으로도 generic surprise/world-model 결합을 신규 개념이라 부르기 어렵다는 판단은 충분하다. 단순히 최신이라는 이유로 이쪽으로 pivot할 근거는 없다.

## 좁은 후속 가설: surprise의 크기가 아니라 행동 수정의 필요성

**가설:** 예측했던 정보와 실제 관측의 차이에서 **action ranking을 바꾸어야 하는 mismatch**를 학습하면, 동일 실제 관측을 그대로 받는 generic-Q보다 noisy/unexpected evidence에서 더 잘 회복한다.

최소 구성은 frozen 관측 encoder, 작은 action-conditioned latent predictor, 작은 correction head다. 예상은 **행동과 인자를 확정한 뒤 실제 결과를 받기 전에 저장**한다. 실제 결과를 본 뒤 만든 예상을 surprise라고 부르지 않는다. tool type만 conditioning하면 서로 다른 검색어 결과가 혼합되므로 검색어/ID 인자도 입력에 포함한다. prediction 이후 실제 관측이 도착하면 residual과 history로 다음 Q를 보정한다. 환경의 full observation text/image를 생성하지 않는다.

학습 supervision은 실제 state의 branch rollout returns에서 얻은 '예측 상태에 따른 선택이 actual state에서 얼마나 regret를 만드는가' 또는 action-value/ranking correction이다. 임의 encoder 거리 자체를 reward로 주는 방식과 분리한다. `Q(h,z_actual)` 대비 `Q(h,z_pred)` 차이를 같은 Q가 스스로 target으로 만드는 경우 self-confirming shortcut이 생길 수 있으므로 독립 actual-return 평가가 필요하다. oracle relevance/gold answer/gold support는 오프라인 label 계산에만 쓰며 input에 넣지 않는다.

중요한 공정성: **동일한 actual observation을 받은 충분한 generic-Q는 residual을 history에서 스스로 계산할 수도 있다.** 따라서 prediction/residual은 새로운 외부 정보가 아니라 inductive bias 또는 training signal이다. generic-Q만 정보가 적거나 proposed만 더 큰 history·학습량을 받으면 실험은 실패다. post-observation Q vector를 입력에 추가하는 단순 generic feature baseline도 포함한다.

최초 pilot은 RAG의 관측 단위에서 충분하다. 이후 **WebShop의 상품 속성/availability 정보**처럼 정답 생성과 다른 실제 선택 문제가 있는 시뮬레이터에서 같은 objective를 평가한다. horizon·action schema는 task별로 고정하고 공통 learning rule을 재사용한다. 두 benchmark family를 학습한 경우 'RAG에서 zero-shot transfer'라고 부르지 않는다. 진짜 transfer는 RAG에서 fixed objective/hyperparameters를 선정한 뒤 WebShop의 별도 train split에서 적응하고 unseen tasks에서 평가하는 **방법의 일반화**부터 주장한다.

## 결정적 실험과 실패 조건

1. **큰 오류·무관 정보:** 내용과 무관한 랜덤 ID/문구 또는 이미지 배경 변화. raw error는 크지만 최적 action/return은 유지해야 한다.
2. **작은 오류·중요 정보:** yes/no 관계, 숫자 하나, 상품 옵션·재고처럼 small perturbation이 최적 행동을 바꾸는 예. 모델은 magnitude가 작아도 필요한 correction을 해야 한다.
3. **예측 가능한 중요 사실:** surprise가 낮지만 계속 retrieval/최종 결정을 돕는 정보. novelty가 낮다고 버리면 실패다.
4. **동일 총오류·다른 관련성:** DRPE의 iso-error 원칙을 natural-language/멀티모달 관측으로 옮겨 norm/length/error magnitude를 맞춘다. 이것이 새 원리가 아니라 기존 진단을 agent로 확장한 것임을 명시한다.
5. **날조·지연·도구 오류:** unexpected observation은 유익한 발견뿐 아니라 실패/노이즈일 수 있다. 높은 surprise를 더 많은 탐색의 긍정 보상으로 무조건 쓰지 않는다.

필수 비교는 같은 데이터·hidden width·학습 epochs의 **generic outcome-Q**, `Q+raw residual magnitude`, `Q+RND novelty`, `Q+predictive uncertainty/disagreement`, **value-aware predictor baseline**, proposed correction이다. 'uncertainty baseline'을 엄밀한 Bayesian IG라고 부르지 않는다. IG baseline을 구현하려면 belief/posterior와 approximation을 명시하고 추가 compute를 기록한다. old access-auxiliary는 선택적 경쟁 가설로 독립 비교하며 기본 모델에 둘 다 넣지 않는다.

보고할 것은 heldout action regret, action-ranking repair, unnecessary switch/overreaction, task success, 반복·실패 후 회복, p50/p95 total cost다. prediction MSE 감소나 surprise-success correlation만으로 decision 품질 개선을 입증하지 못한다. 작은 controller를 위해 actual image를 매번 full VLM으로 추가 encoding하면 비용 절감 주장은 철회하고 품질 이득만 검토한다.

**기각 조건:** nuisance 변화에도 행동을 과도하게 바꾸거나, task-critical 작은 변화에 무반응이거나, 동일정보 generic-Q/value-aware baseline을 이기지 못하거나, RAG 외 family에서 효과가 사라지면 이 가설을 지지하지 않는다. learned correction 대신 선형 residual feature만으로 같은 결과가 나와도 복잡한 world-model 모듈은 기여에서 제외한다.

## 논문에서 배울 것과 현재 코드 점검

| 출처 | 배울 원칙 | 현재 확인할 코드 경계 | 별도 실험 / 우선순위 |
|---|---|---|---|
| ICM/RND | prediction error와 novelty의 서로 다른 source | 현재 DirectHFReader에는 둘 다 없음; 있다고 설명하지 않음 | raw residual/RND 저비용 대조 / P0 |
| VIME/ICL-curiosity | error≠epistemic gain; noise 분리 | action 이전 forecast와 이후 observation timestamp 분리 | noise/distractor stress tests, calibrateddisagreement / P0 |
| DRPE/VaGraM/value-equivalence | task 영향과 observation fidelity 분리 | full registry action과 실제 handler/returns 동일성 | iso-error 및 generic value-aware baseline / P0 |
| AEWM/Qwen-AgentWorld | tool text 복원보다 action 개선을 직접 평가 | arg JSON generation·full prefill 여전히 존재 | semantic forecast utility와 actual action repair / P0 |
| JEPA/Dreamer | latent prediction은 원리, 필요한 학습 규모는 별개 | Jev 이름과 JEPA objective 혼동 금지 | pretrained frozen embeddings로 작은 feasibility부터 / P1 |
| AWM | 비RAG executable transitions가 필요 | 실제 외부 side effects 없는 simulator 선택 | WebShop 또는 제한된 AWM tasks로 원리 재검증 / P1 |

**진행 권고:** 기존 access-auxiliary를 확정 논문 기여로 유지한 채 surprise를 덧붙이지 않는다. 먼저 **같은 actual observations에서 generic-Q 대비 prediction mismatch가 action repair를 설명하는지** small diagnostic을 한다. 신호가 강할 때만 surprise/correction을 중심 가설로 **교체**한다. 신호가 약하면 access-effect 쪽으로 돌아가며, 두 약한 아이디어를 합쳐 기여처럼 만들지 않는다.

## 최종 사용자 설명 반영: 중심은 사후 surprise가 아니라 행동 전 imagination planning

위의 action-repair pilot은 post-observation 부품에 치우쳐 있다. 최신 의도에 맞춘 **주된 연구 후보는 다음 전체 한 가지 모델기반 planning 가설**로 수정한다.

`현재 관측 이력 → 합법 행동·인자 후보 → 예상 다음 관측/state latent → 짧은 lookahead의 예상 최종 task utility−cost → 하나를 실제 실행 → 실제 관측으로 state/prediction 갱신 → 반복`

surprise는 이 루프의 forecast correction 신호다. intrinsic novelty reward를 별도 목표로 추가하는 것은 기본 설계에서 제외한다. 직접 scalar Q만 예측하면 generic value controller이며 world model을 구현한 것이 아니다. **action-conditioned 다음 관측 또는 latent state를 실제로 예측하고 그 predicted state로 이후 action/terminal value를 계산**해야 모델기반 주장을 할 수 있다. original A-RAG는 대조 기준의 성능이며 수학적 upper bound는 아니다. 더 좋은 선택을 학습하면 이를 넘는 것도 목표다.

### 이 해석에서 더 직접적인 기존 planning 연구

- [**Reasoning with Language Model is Planning with World Model (RAP)**, 2023-05-24](https://arxiv.org/abs/2305.14992): LLM을 world model과 reasoning agent로 사용하고 MCTS로 미래 states/rewards를 탐색한다. '예상 결과를 보고 더 좋은 action을 고른다'는 큰 원리는 이미 직접 존재한다. 우리 후보는 full textual imagination 대신 저비용 latent transition으로 agent tool outcomes를 모델링하고, 실제 return/regret로 그 예측을 검증하는 데 차이가 있어야 한다.
- [**Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models (LATS)**, 2023-10-06 / ICML 2024](https://arxiv.org/abs/2310.04406): LM value/reflection과 tree search를 external environment feedback에 결합한다. 실제 환경 branch를 실행하는 검색과 학습된 world model 안의 imagined branch는 비용과 오류 특성이 다르다. budget-matched 작은-depth search 비교가 필요하다.
- [**DynaWeb: Model-Based Reinforcement Learning of Web Agents**, 2026-01-29](https://arxiv.org/abs/2601.22149): action-conditioned web page world model에서 imagined rollout을 만들고 expert/online trajectories와 섞어 agent policy를 학습한다. 비RAG world-model agent 학습·일반화 자체도 선점된 영역이다. inference-time short lookahead와 offline world-model training의 역할을 분리해야 한다.
- Dreamer/TD-MPC2/value-equivalence와 위 Qwen-AgentWorld는 latent planning·가치보존·예상 feedback 기반 refinement의 더 넓은 비교 계열이다. **작은 latent predictor+lookahead라는 구조만으로 독창성이 확정되지 않는다.**

### 제안하는 가장 작은 실험

첫 후속 실험의 유일한 조작은 **같은 frozen reader/같은 관측 state/같은 합법 candidate actions와 arguments를 받는 direct-Q 대신 learned latent transition으로 짧게 lookahead했을 때 더 좋은 선택을 하는가**다. 세 depth를 사전 고정한다: depth0=generic Q, depth1=next-state prediction+terminal value, depth2=한 번 더 latent transition과 decision. deeper search·memory·CoT compression·access-auxiliary를 한꺼번에 더하지 않는다.

후보 action에 query string 또는 figure/chunk IDs가 필요한 경우 두 가지 중 하나를 실험 전에 선택한다. (a) 같은 고정 argument generator로 후보별 인자를 먼저 만들고 **모든 후보 생성 비용**을 지불한다. (b) `P(u|h,a)`에 대한 expected utility를 추정하고 sampling/marginalization 비용을 공개한다. `F(h,a)`만으로 실제 서로 다른 query 결과를 예측하면서 argument 비용이 없다고 가정하지 않는다. 제한된 pointer 후보는 가능한 별도 task restriction이지만 기존 full A-RAG 자유 검색과 동일 과제로 포장하지 않는다.

이미지 후보의 **아직 열어보지 않은 pixels를 predictor input에 넣지 않는다**. 공개 retrieval metadata와 이미 계산해둔 index embeddings를 사용하면 그 제한된 정보만으로 예측한다. 실제 이미지 read 뒤에 얻는 VLM state와 비교하면 별도의 post-action target이며 비용도 실제로 발생한다. pre-read 예상과 post-read actual이 같은 encoder coordinate system인지 확인하고, stochastic/다중 가능한 결과에 deterministic mean latent만 쓰면 비실재 상태가 생길 수 있음을 평가한다.

primary outcome은 최종 task success/EM/F1와 **실제 branch outcome에 대한 action-ranking regret**다. secondary는 task-relevant predicted state fidelity/calibration, rollout depth별 model error, false optimism과 value overestimation이다. depth2가 오히려 악화되면 상상 오류 누적을 숨기지 않는다. 다음 controls가 핵심이다: matched genericQ, model없이 더 큰 MLP, shuffled transition predictor, actual-state/oracle one-step lookahead(진단 ceiling), raw-surprise bonus, disagreement/uncertainty, task-value-aware transition loss. direct-Q와 world model은 같은 transition/return data와 comparable parameter budget을 받는다.

**가설 지지 조건:** 실제로 좋은 tool choice의 ranking을 더 잘 예측하고, 그 ranking 개선이 frozen reader의 최종 성공 개선으로 이어지며, 동일 연산/상호작용 예산에서 성립해야 한다. observation prediction MSE만 개선하거나 extra rollout/argument candidates 때문에 좋아진 결과는 불충분하다. RAG에서 시작해 WebShop 같은 비RAG agent family에서 같은 transition/value objective를 고정해 검증한다. 이전 access-auxiliary는 이 가설에 자동으로 결합하지 않는다. 새 pilot이 genericQ를 이길 때에만 기존 중심 가설을 교체한다.
