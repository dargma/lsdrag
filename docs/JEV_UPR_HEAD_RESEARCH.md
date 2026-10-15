# UPR를 참고한 JEV head와 도구 인자 생성 축소안

작성 2026-10-15 UTC. 연구 검토이며 구현·성능 결과가 아니다. 현재 진행 중인 고정 실험의 action/kernel/데이터를 변경하지 않는다.

## 확인한 UPR의 의미

여기서 UPR는 Sachan et al., EMNLP 2022의 **Unsupervised Passage Re-ranking**이다. [원 논문 §2.2](https://aclanthology.org/2022.emnlp-main.249.pdf)는 후보 passage c마다 이미 주어진 질문 q의 평균 token log-likelihood를 계산한다.

`r(c,q) = (1/|q|) Σ_t log p(q_t | instruction(c), q_<t)`

새 질문을 실제로 샘플링하거나 autoregressive decode해서 점수를 얻는 방식이 아니다. 입력 질문을 teacher forcing하고 target 위치의 확률을 읽는다. 따라서 제목의 “Question Generation”은 likelihood의 방향을 뜻하며, 온라인 생성 문장 수가 후보 수만큼 늘어나는 구현을 요구하지 않는다. 단, 후보마다 다른 passage-conditioned forward는 필요하다. 후보가 K개면 K개 조건부 평가를 수행하며 batching은 이 계산을 무료로 만들지 않는다.

저자 [공식 저장소](https://github.com/DevSinghSachan/unsupervised-passage-reranking)의 확인 revision은 `a05d296fc8c5cc19cbf769a8f9c5d76b4d002b25`다. [upr.py 135–149행](https://github.com/DevSinghSachan/unsupervised-passage-reranking/blob/a05d296fc8c5cc19cbf769a8f9c5d76b4d002b25/upr.py#L135-L149)은 labels에 질문을 넣은 forward→log_softmax→target gather→후보 정렬이다. 코드의 sum과 논문의 mean은 같은 질문 토큰 수로 모든 후보를 비교할 때 순서가 같다. [GPT 구현 173–188행](https://github.com/DevSinghSachan/unsupervised-passage-reranking/blob/a05d296fc8c5cc19cbf769a8f9c5d76b4d002b25/gpt/upr_gpt.py#L173-L188)도 shifted supplied labels에 CE를 적용하며 generate()로 후보 질문을 만드는 경로가 아니다.

로컬 IRM 초안의 식 rescore와 측정 설명도 같은 의미다. 확인 파일은 `/content/drive/MyDrive/IRM_ICLR_2027_darft_ver2.1.tex` 156–162행 및 711행 이후, bib는 `/content/drive/MyDrive/IRM_ICLR2027_current/refs.bib`의 `sachan2022upr`다. IRM 설명의 “필요한 질문 위치에만 LM head 적용”은 passage 위치의 vocabulary projection을 줄이는 최적화다. **질문 위치에서도 전체 vocabulary의 logsumexp 분모가 필요**하므로 target row dot product만 구하고 정확한 log-probability라고 할 수 없다. Transformer forward와 후보별 비용도 남는다. 이번 검토에서는 IRM 실행 코드의 동치성과 실측을 새로 인증하지 않았다.

## 현재 목표에 옮길 수 있는 부분

UPR의 유용한 교훈은 “모델이 문자열을 출력하도록 하지 않고, 필요한 scalar/선택을 이미 계산한 표현에서 읽는다”이다. 그러나 UPR 자체는 행동 utility learner나 멀티모달 controller가 아니다. `p(question|passage)`와 “이 항목을 다음에 읽으면 최종 QA가 개선되는가”는 서로 다른 target이다.

기존의 도구 호출이 `{"tool":"read","id":"long_document_id"}`를 생성한다면, head가 유효 action과 local candidate index를 직접 선택하고 Python adapter가 실제 함수와 ID를 매핑할 수 있다. 이때 **tool 이름·ID·JSON의 decode**가 없어지는 것이며, 모델 실행 자체나 읽지 않은 후보의 이해가 무료가 되는 것은 아니다.

## 비교할 세 가지 설계

| 설계 | 온라인 경로 | 학습/점수 | 비용과 주의점 |
|---|---|---|---|
| 1. 상태 action head + candidate pointer | 관측 상태의 frozen hidden h에서 STOP/TEXT/IMAGE/READ를 선택하고, 이미 검색된 shortlist의 후보 표현 z_i에 shared score `vᵀ tanh(W_h h + W_z z_i + b)`를 적용해 index를 반환 | 같은 branch return 또는 제한된 imitation으로 action과 target 학습 | 가장 작은 증분. 동적 후보 ID를 출력 class로 외우지 않는다. 후보 표현이 retriever embedding/제목이면 pixel 이해는 없으며, raw 이미지를 전부 VLM에 넣으면 image cost를 이미 지불한다 |
| 2. 유효 action–target pair의 공동 점수 | `{STOP} ∪ {(READ_TEXT,i),(READ_IMAGE,j),...}`에 `Q(h,a,z_i)`를 부여하고 legal mask 후 하나를 선택, 직접 dispatch | 실제 downstream return에서 공동 utility 학습; 별도 head보다 action/target 불일치 감소 가능성 검증 | 후보 수·modality별 점수 scale 및 cardinality bias를 통제한다. TEXT_SEARCH(new query)는 기존 문서 pointer와 의미가 달라 query 생성 또는 query 후보 집합이 별도로 필요 |
| 3. UPR teacher→작은 pointer distillation | 학습 때만 후보별 UPR scores를 계산하고 shared pointer가 teacher ranking을 예측; 배포 때 teacher 미실행 | `KL(softmax(r_UPR/τ) || softmax(r_head/τ))` 같은 relevance 보조 loss를 실제 return loss에 추가 | 가장 무거운 후보별 teacher를 오프라인으로 옮길 가능성. 하지만 관련성 teacher는 bridge evidence를 낮게 평가할 수 있다. UPR-only imitation을 주 방법으로 채택하지 않고 downstream utility와 비교 |

후보 위치를 출력하는 발상 자체는 기존 [Pointer Networks](https://papers.nips.cc/paper/2015/hash/29921001f2f04bd3baee84a12e98098f-Abstract.html)와 넓게 겹친다. 작은 pointer·joint head·distillation을 조합했다는 사실만으로 새 논문 기여라고 주장하지 않는다. 핵심 가설은 기존 제안의 접근 대조 supervision이 이러한 저비용 후보 선택에서도 동등 예산 일반 value learner보다 유용한가이다.

첫 구현 후보는 **설계 1**이다. 현재 head가 아직 검증되지 않았으므로 설계 2·3을 동시에 추가하지 않는다. 기존 단계 1/2 결과를 고정한 뒤, 인자 생성 비용이 실제로 큰 경우에만 별도 버전에서 비교한다.

## 모든 설계가 지켜야 하는 경계

- **후보 coverage:** pointer는 후보에 없는 문서·새 검색어를 선택할 수 없다. 모든 controller가 동일 shortlist를 받아야 하며, 후보 수 K와 recall–latency curve를 보고한다. 기존 top-1 acquisition을 top-K 선택으로 바꾸면 acquisition kernel 자체가 바뀌므로 기존 controller 실험과 분리한다.
- **새 query 생성:** 원 질문 재사용, 관측 entity span, 고정 rewrite bank는 문자열 생성을 줄일 수 있지만 탐색 공간을 제한한다. 새로운 compositional query가 필요하면 생성 경로를 남기고 그 비율·비용·품질을 함께 측정한다. 오프라인으로 미리 생성한 query bank 비용도 기록한다.
- **읽지 않은 후보와 미래 정보:** pointer 입력에는 호출 전에 합법적으로 존재하는 후보 metadata/표현만 넣는다. unexecuted tool result, 미래 answer, gold support/UPR teacher target을 runtime feature로 넘기지 않는다. 이미지 UPR는 원 text UPR의 검증 결과가 그대로 적용되지 않는 별도 VLM 확장이다.
- **KV·position:** 같은 prefill에서 얻은 h와 KV를 재사용할 수 있어도, 텍스트로 직렬화한 tool call을 Python dispatch로 없애면 다음 turn history·token positions가 바뀐다. canonical synthetic tool event를 모든 비교군에서 동일하게 넣거나, 새 history의 시스템 효과라고 명시한다. 기존 prefix KV에 다른 token/history를 억지로 붙이지 않는다. iteration 간 cache 재사용도 별도 계측한다.
- **확률 해석:** UPR likelihood softmax는 후보 내부 상대 점수, Q-softmax는 선택 preference다. 둘 다 정답 확률이나 개선 확률로 자동 해석하지 않는다. 후보 수·순서 변화, OOD 문서, tool budget에 대해 calibration/선택 안정성을 검증한다.
- **표현 비용:** 후보 representation 생성·pooling·복사·GPU↔CPU 전달까지 포함한다. 동일 prefill에 후보를 추가하면 prefix와 image 처리 비용이 증가한다. 작은 head FLOPs만 제시하지 않는다.

## 검증할 최소 비교와 판정

동일 backbone·입력·shortlist·history·query/answer 생성·tool budget에서 (A) 기존 생성형 tool+ID, (B) constrained first-token action 및 생성형 ID, (C) action head 및 생성형 ID, (D) action head 및 pointer ID를 비교한다. ID 선택만 바뀐 forced-action replay도 수행한다. UPR teacher 도입은 별도 (E) pointer+UPR 보조학습으로 두며, 같은 label GPU-hour의 추가 return-data baseline과 비교한다.

측정은 tool명/JSON/ID decode tokens, invalid-ID/fallback, 선택 일치율, candidate recall, end-to-end QA–latency, 평균/p50/p95, 후보 표현·teacher 수집 비용이다. 절감 가능한 상한은 먼저 현재 trace의 **도구 호출 직렬화와 인자 decode가 차지하는 비율**로 계산한다. 이미지 prefill/최종 답변이 지배적이면 pointer 개선의 시스템 효과는 작을 수 있다.

채택 조건은 동일 품질·예산에서 실제 전체 지연 감소가 확인되고, 잘못된 target 선택이나 새 query 부재로 생긴 실패가 이득을 상쇄하지 않는 것이다. UPR distillation이 ranking만 개선하고 폐루프 QA–latency에 이득이 없으면 controller 방법으로 성공했다고 판단하지 않는다. 현재 이 문서의 어느 설계도 실행·검증되지 않았다.
