# Head 구조 가속: 직접 실행, vocabulary projection 생략, 후보 pointer

2026-10-15. 사용자의 구조 의견과 후속 조사·Astra 검토 요청을 반영한 계획이다. 기존 v1.1 수집 조건은 바꾸지 않는다. 검토를 통과한 추가 arm만 별도로 실행한다.

## 1. 사용자 의견과 현재 구현의 구분

사용자가 원하는 구조는 head가 행동을 고른 뒤 다시 도구명·ID 문자열을 생성하게 하지 않고, 선택 결과를 실행 함수에 직접 연결하는 것이다. 유한한 후보가 있으면 인자도 해당 후보의 index로 선택한다.

```text
관측된 상태 → 표현 → 행동/대상 점수 → legal (action, slot)
                                      ↓
                       runtime.dispatch(action, candidates[slot])
```

v1.0은 행동 head 뒤 native tool prefix를 같은 prefill KV에 붙여 인자를 생성했다. 도구 선택의 자유 생성은 줄였지만 full VLM prefill·forced-prefix forward·인자 decode는 남았다. 현재 v1.1은 이미 action index에서 Python acquisition/answer 함수로 바로 분기한다. **현재 v1.1에는 도구명·문서 ID·tool JSON 생성이 없으므로, 이를 새롭게 없앴다는 가속 주장은 성립하지 않는다.** 남은 주요 생성은 실제 새 검색어와 최종 답변이다.

## 2. UPR에서 가져올 원리와 가져오면 안 되는 가정

IRM의 `sachan2022upr` 인용은 [Improving Passage Retrieval with Zero-Shot Question Generation](https://aclanthology.org/2022.emnlp-main.249/)이다. UPR는 주어진 질문 q를 후보 c에 조건화한 평균 teacher-forced log-likelihood로 재정렬한다. 질문을 새로 샘플링하는 작업은 아니다.

그러나 후보별 contextual forward와 질문 위치의 정규화된 vocabulary likelihood 계산은 필요하다. 전체 vocabulary 분모 없이 정답 token weight row만 dot product해서 정확한 log-probability라고 할 수 없다. 반면 제한된 행동 A/B/C의 **조건부 상대 분포**만 필요하면 해당 세 row의 logits로 softmax를 계산할 수 있다. 두 목적은 다르다.

UPR의 관련성 점수는 현재 다음 행동의 최종 QA 개선량과 동치가 아니다. UPR를 online에서 모든 후보에 실행하면 새 query 생성보다 더 비쌀 수도 있다. 활용 후보는 작은 shortlist의 명시적 baseline 또는 offline ranking teacher이며, 후자는 실제 return 학습과 별도 ablation을 요구한다. [UPR 조사](JEV_UPR_HEAD_RESEARCH.md)

## 3. 지금 실행할 수 있는 동치 가속 arm

현재 `V11Backend.encode_state()`는 full conditional-generation wrapper에서 마지막 위치의 전체 vocabulary logits까지 계산한 후 hidden을 반환한다. 학습된 Q head는 그 vocabulary logits를 사용하지 않는다.

추가 arm은 **동일한 전처리·이미지·PLE·전체 multimodal backbone·prefix를 그대로 실행하되, 마지막 vocabulary projection만 생략하고 동일한 norm hidden을 읽는다.** 이는 모델 층을 줄이거나 embedding으로 이미지를 대신하는 것이 아니다. Q head와 Python dispatch는 기존과 동일하다. 새로운 학습 target이나 candidate kernel이 필요하지 않다.

같은 원리로 first-token baseline은 전체 LM head 대신 A/B/C row와 bias(있는 경우)를 선택해 projection하고, 같은 Gemma logit softcapping을 적용한다. FP32 3-way softmax는 현재 A/B/C에 조건화한 상대 분포와 수학적으로 같다. 다만 다른 크기의 BF16 GEMM이 rounding이나 near-tie argmax를 바꾸는지는 실제 GPU로 검사해야 한다.

검증 순서:

1. 승인된 수집이 끝나 GPU를 순차 사용한다. 고정 train probe의 text/image/표 입력에서 기존 full-wrapper와 backbone-only hidden을 비교한다.
2. 동일하게 학습한 head의 Q·action 일치, first-token full vs selected-row action·분포 차이를 보고한다. 불일치를 평균으로 숨기지 않는다.
3. 동치가 확인된 학습 head arm을 development에 추가한다. first-token도 동등 최적화를 제공하거나 기존/최적화 지연을 함께 보고한다.
4. warmup 이후 교차 실행 순서의 paired latency를 비교한다. preprocessing·CPU 전달·head 추론·최종 QA까지 비용을 포함한다.
5. 효과가 작으면 작은 engineering 이득으로 보고하며 새로운 논문 알고리즘이라고 부르지 않는다.

prefix가 바뀐 query/answer로 임의 KV를 이어 붙이는 변경은 하지 않는다. 공통 prefix cache를 추가하려면 토큰 단위 prefix identity와 position semantics를 검증하는 별도 arm이 필요하다. 이번 최적화는 iteration 간 cache 재사용을 달성했다는 의미가 아니다.

## 4. 더 큰 구조 후보: action+target pointer

상태 h와 runtime 후보의 표현 z_i에서 `(READ_TEXT,i)`, `(READ_IMAGE,j)`, `ANSWER_AND_STOP`, 필요한 경우 `NEW_QUERY`를 점수화하고 유효 pair를 직접 실행하는 안이다. 후보 ID 문자열을 class로 외우지 않고 shared score 함수로 가변 후보 index를 고른다.

UPR posterior는 ranking auxiliary 또는 teacher로 검토할 수 있으나, STOP과 비교할 점수는 같은 downstream utility 척도로 맞춰야 한다. 관련성 softmax를 정답/종료 확률과 바로 비교하지 않는다. 후보 표현·pooling·cache·후보 생성 비용도 계측한다.

**현재 수집한 modality Q3 label은 candidate별 label이 아니다.** pointer는 후보집합과 acquisition kernel을 바꾸므로 현재 head를 그대로 붙여 동등 비교라고 할 수 없다. 정확한 candidate coverage, same-candidate first-token 대조군, 실제 candidate-conditioned returns가 준비되기 전에는 현재 실험 arm으로 채택하지 않는다. 새 query를 만들 수 없는 제한도 명시한다. [구조 조사](JEV_DIRECT_DISPATCH_HEAD_DESIGN.md)

cached-state pointer가 VLM controller prefill까지 생략할 수 있는지는 추가 연구 질문이다. embedding에 시각 세부 정보가 충분하다는 보장은 없고, REFRESH/NEW_QUERY로 회복할 수 있는지와 그 비용을 함께 검증해야 한다. 이것을 단순한 현재 head의 속도 최적화와 혼동하지 않는다.

## 5. 실행 범위 결정

현재 채택 후보는 **same-prefix backbone-only head + 공정한 selected-row first-token 비교**다. 조건은 실제 hidden/action 동치와 측정된 지연 이득이다. 학습된 value/access head의 과학적 비교는 유지한다.

UPR/pointer/cache-state는 구조 의견에 대한 연구 후보로 문서화한다. 이들을 구현해 좋은 결과를 얻었다고 주장하지 않으며, 새로운 후보별 supervision 없이 현재 target을 재활용하지 않는다. [Astra 검토](JEV_HEAD_ACCELERATION_ASTRA_REVIEW.md)의 최종 조건에 따라 추가 arm을 고정한다.
