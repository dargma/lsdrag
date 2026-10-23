# 같은 LLM의 decision head: 최신 직접 선행 점검

조사일·cutoff: 2026-10-23. 기존 문헌 메모는 검색 단서로만 사용하고 아래 원문을 다시 확인했다. 논문 수치의 재현 실험은 하지 않았다. 모델 저장소의 생성일은 발표일과 같다고 단정하지 않는다.

**판단:** frozen hidden state에 작은 head를 붙이는 것, 선택 확률을 출력하는 것, 같은 모델이 선택과 생성을 맡는 것 각각에는 직접 선행이 있다. 우리 후보 차이는 고정된 외부 도구 계약과 non-thinking 생성기를 유지하면서, 실제 후속 trajectory의 답변 품질·남은 비용으로 모든 단계의 작은 정책을 학습하고 동일 prefill/cache에서 인자·답변 생성을 이어가는 데 있다. 이 결합도 최초라고 주장하지 말고 강한 비교 실험으로 효용을 입증해야 한다.

## 원문으로 확인한 6개 모델·연구

| 연구 / 확인 날짜 | 원리·학습 신호 | 우리 실험과의 관계 |
|---|---|---|
| **FunDM**, 2026-10-12, arXiv v1 | frozen Qwen3-8B 중간층의 각 option 끝과 answer 위치를 함께 읽는다. 128차원 query/key dot product 또는 작은 transformer 1층을 추가한 head만 KL 학습. ARC/SciQ/MedMCQA 38,067행에서 full-model 선택 확률을 증류하며 정답 라벨은 fitting에 쓰지 않는다. | 위치별 readout과 작은 구조 비교의 직접 선행. 완성된 coding run을 고르는 별도 head 실험도 있지만, 환경 안에서 매 단계 행동을 바꾸는 terminal-return 정책은 아니다. [논문 §3–4](https://arxiv.org/pdf/2610.17175), [저자 코드](https://github.com/Ujjawal-K-Panchal/FunDM) |
| **OLIVIA**, 2026-05-11, arXiv v1 | frozen 마지막 hidden을 context로 쓰는 action별 linear bandit. 같은 모델의 action-name embedding으로 초기화하고 LinUCB 및 rank-one 통계 업데이트. ToolBench/TaskBench/TaskBench-MM/BFCL의 reference tool multiset에 남아 있는 tool인지 즉시 binary feedback을 받는다. | frozen LLM+명시적 학습 decision layer의 매우 가까운 선행. 사고문 생성은 유지하며, oracle tool 수 기반 budget도 사용한다. 우리의 최종 QA·미래 비용 target과 다르다. 같은 label budget의 bandit/회귀 비교를 고려할 근거다. [§4–6, Algorithm1](https://arxiv.org/html/2605.11169v1) |
| **LAnR**, 최초 2026-04-20, **최신 v3 2026-10-19** | 같은 LLM의 `[PRED]` hidden이 dense query와 retrieve/stop을 결정하고 최종 답변도 생성한다. 본문은 MLP라 부르지만 B.1 구현 명세는 affine+sigmoid, 추가 LLM forward 0. NQ+HotpotQA 169,615예제로 backbone 전체를 contrastive+NTP+control BCE 학습한다. 정지 label은 gold evidence 회수 여부다. | 동일 LLM 선택·생성 및 작은 정지 head의 직접 선행. 우리의 backbone frozen/복수 typed tool/후속 QA-cost 학습과 구분된다. **v2만 인용하던 기존 메모를 최신화해야 한다.** [v3 §3/B/B.1](https://arxiv.org/html/2604.17866v3), [버전 이력](https://arxiv.org/abs/2604.17866) |
| **D3-Flash**, HF repo 생성 2026-10-10, 확인 revision 13dc1cc | Qwen3.5-9B 계열과 answer-code readout. 질문별 forward에서 마지막 위치로 선택지를 채점하고 text generation 없이 확률을 반환한다. 공식 serving recipe는 full-attention의 양방향 처리 때문에 prefix cache/chunked prefill을 끈다고 명시한다. | 확률형 decision API 비교 대상이지만 별도 결정 backbone이다. 현 causal generator의 KV 연속 생성과 실행 조건이 다르다. **확인한 card/recipe에 전체 훈련 데이터·loss recipe는 없으며 RL이라고 단정 불가.** [card](https://huggingface.co/vllm-sr/d3-flash), [공식 recipe](https://recipes.vllm.ai/vllm-sr/d3-flash) |
| **D3-Turbo**, HF repo 생성 2026-10-19, 확인 revision a0f3e9e | 공개 card상 27B multimodal decision 모델이며 요청 state를 한 번 읽어 여러 typed 질문을 답한다. 선택·yes/no·score 이외 위치/track 형식도 제공한다. 전체 학습 mixture·objective는 이번 확인 범위에서 미확인이다. | Flash의 질문별 forward와 혼동하지 말 것. 같은 state의 여러 판단 비용을 줄이는 설계 선행이지만 우리가 생성하는 검색어·답변을 대체하는 단일 generator가 아니다. [card](https://huggingface.co/vllm-sr/d3-turbo) |
| **Decision-1.0-Route-0.6B**, HF repo 생성 2026-09-28, 확인 revision 6c59e23 | Kai의 Choice/Noul 경로를 request-time 10개 신호에 맞춰 fine-tune. 공개 training record는 distinct 366,323행, encoder/head LR, task mixing과 dev NLL 선택을 설명한다. Toucan/glaive tool need, RAGBench hallucination 등의 데이터가 포함된다. encoder도 업데이트하며 head-only가 아니다. | 명확한 데이터 provenance의 참고 대상. D3 훈련 데이터라고 전용하면 안 된다. 요청 분류·tool 필요성 label은 미래 trajectory 가치와 다르다. dev temperature/threshold가 새 traffic으로 그대로 전이되지 않는다고 저자도 명시한다. [card](https://huggingface.co/vllm-sr/Decision-1.0-Route-0.6B), [training record](https://huggingface.co/vllm-sr/Decision-1.0-Route-0.6B/blob/main/TRAINING_DATA.md) |

## 현재 설계에 주는 구체적 의미 — 조사자의 추론

- **표현·구조:** 마지막 token 대 모든 token attention만 비교하기보다, 실제 후보/도구 설명 끝과 현재 answer 위치를 읽는 option-aware readout이 설득력 있는 후보다. FunDM 구조를 그대로 재현한 baseline과 우리 구조를 구분해야 한다. 중간층 early exit는 뒤의 생성에 full-layer KV가 필요하므로 같은 비율의 end-to-end 절약을 약속할 수 없다.
- **학습:** teacher 확률 KL, gold evidence sufficiency BCE, immediate tool reward, terminal QA-minus-cost는 서로 다른 supervision이다. 동일 수집 예산과 동일 state에서 비교해야 어떤 개선이 실제 결과 기반 학습 덕분인지 판단할 수 있다.
- **확률:** categorical softmax, LinUCB uncertainty, 성공확률은 같은 것이 아니다. 현재 `action_probabilities`는 선택분포다. calibrated success/value 출력이 필요하면 실제 outcome target과 별도의 holdout calibration을 설계해야 한다.
- **효율:** same-prefill head 자체는 LAnR 선행이므로 새로움의 근거가 약하다. syntax 공급·KV 재사용을 Native에도 제공한 조건에서 learned selector의 추가 이득과 offline 학습비용을 보고해야 한다.
- **현재 범위:** 이 문서는 연구 비교다. 진행 중인 pilot2의 split, objective, 하이퍼파라미터를 바꾸거나 추가 실험을 실행하지 않았다. FunDM/OLIVIA/LAnR 재현 성능은 미확인이다.

## 검증 깊이와 남은 불확실성

FunDM PDF §3–4와 옵션/학습 설명, OLIVIA HTML §4–6·Algorithm1, LAnR v3 §3·B·B.1을 읽었다. OpenReview OLIVIA 링크는 browser challenge라 arXiv 원문으로 대체했다. D3는 저자 card와 공식 runtime recipe, Decision1은 card+TRAINING_DATA를 확인했다. HF API의 createdAt/lastModified/revision을 따로 기록했다. 출처별 원문 위치·미확인 항목은 [head_sources.json](../../../lab/reports/related_work_20261023/head_sources.json)에 있다. 전체 최신 문헌의 망라나 novelty 증명은 아니다.

## 벤치마킹해서 가져올 방법 2개 — 다음 실험 제안, 아직 미실행

**1. OLIVIA에서 가져올 부분: 작은 linear reward predictor와 탐색 불확실성의 분리.** 충실 재현은 reference tool multiset, 원 논문의 thought 생성, per-action 통계·초기화·UCB, stream 평가와 oracle budget을 모두 맞춰야 한다. 현재 RAG에 그대로 적용하면 reward 의미와 budget이 달라지므로 재현으로 부르면 안 된다. 우리의 적응형 비교에서는 이미 수집한 동일 state/action/suffix 데이터의 `F1−λ·remaining_cost`를 target으로 삼아 ridge action-value head를 학습하고 argmax한다. 이는 **OLIVIA-inspired offline value regression**이지 원 논문의 online LinUCB가 아니다. 최소 비교는 동일 hidden/상태/분기 데이터/λ/실행기에서 현재 return-PG, outcome-CE, linear value regression 3개다. 각 문항의 학습 기여와 rollout 비용을 맞춘다. 후속 online UCB를 비교하려면 양쪽에 동일 수집 상한을 부여하고 선택확률·reward·episode 순서를 기록해야 하며, 전체 공분산은 action당 d×d 비용이 있어 low-rank화를 추가하면 별도 ablation으로 명시한다.

**2. FunDM에서 가져올 부분: 후보별 readout과 작은 attention scorer.** 충실 재현은 원 lettered MCQ prompt, option 끝/answer 위치, RMS normalization, 128차원 query/key, teacher KL과 원 데이터 split을 맞추는 것이다. 우리의 적응형 비교는 도구 설명 또는 현재 read 후보 끝 hidden을 key, decision 위치 hidden을 query로 쓰는 공용 scorer다. 기존 고정 4-class linear/MLP와 비교하되 backbone·terminal target·질문 split·state 수를 고정하고 head 파라미터 수를 함께 보고한다. 후보 위치가 실제 prompt에 없으면 표식 삽입은 generator 입력도 바꾸므로 **표현 변경과 prompt 변경을 분리한 대조군**이 필요하다. 먼저 마지막 층만 써서 생성용 KV 조건을 유지하고, 중간층 early exit는 별도 연구로 미룬다. teacher-KL 대 outcome target 비교를 따로 해야 구조 이득과 학습 신호 이득을 혼동하지 않는다.

두 제안 모두 현재 pilot2의 노출된 평가 결과를 보고 train hyperparameter를 맞추지 않는다. 구현 전 작은 train-only parity 검사, 학습 예산 동결, 새로운 holdout으로 평가한다. 새 성능·속도·일반화 결과는 아직 없다.
