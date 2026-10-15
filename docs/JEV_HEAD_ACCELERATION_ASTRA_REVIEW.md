# JEV head 가속 재검토: 즉시 적용할 최적화와 후속 구조 실험

작성: 2026-10-15 UTC. 사용자 요청의 head 구조·효율 검토. [UPR 연구안](JEV_UPR_HEAD_RESEARCH.md), [직접 dispatch 설계](JEV_DIRECT_DISPATCH_HEAD_DESIGN.md), 현행 `v11_backend.py`, `run_v11_pilot.py` 및 설치된 Transformers Gemma4 forward 구현을 읽었다. GPU 실험, 학습, 코드 수정, git 변경은 수행하지 않았다. 이 문서의 승인과 실제 동치성·속도 검증 결과는 구분한다.

## 1. 판정과 우선순위

**지금 추가할 실험은 동일 관측 prefix에서 불필요한 vocabulary projection을 제거한 readout이다. CPU/GPU 동치 검증을 통과하면 기존 v1.1과 별도의 추가 arm으로 적용할 것을 승인한다.** 사용자 요청 범위 안의 구체적이고 검증 가능한 최적화다. 가속 크기나 정확도 개선은 아직 확보하지 않았다.

| 순위 | 실험 | 판정 |
|---|---|---|
| 1 | learned head: 동일 multimodal backbone의 마지막 정규화 hidden만 읽고 LM vocabulary head 생략 | **즉시 구현·동치 검증 후 적용 승인**. 행동/kernel/label/feature prefix 유지 |
| 1, 공정성 대조 | first-token: 같은 hidden에서 A/B/C의 LM-head 세 행만 계산 | **함께 적용 승인**. 기존 full-vocabulary first-token과 동치 검증 필요 |
| 2 | 동일 관측의 공통 prefix prefill을 decision/query/answer branch 사이 재사용 | 후속 시스템 실험. 정확한 token prefix·position·attention·multimodal 상태 동치성을 입증한 후 별도 적용 |
| 3 | 공통 shortlist의 full-state action–candidate 공동 head | 별도 탐색 pilot로 조건부 승인. 후보별 downstream labels 새로 수집해야 함 |
| 4 | cached-state pointer + selective refresh; UPR teacher auxiliary | 앞선 후보 coverage와 return signal이 확보될 때만 작은 후속 실험. 현 실험의 즉시 대체 불승인 |

현재 v1.1은 이미 `action index → Python acquire/answer`로 직접 dispatch한다. 온라인에서 tool 이름·document ID·tool JSON을 생성하는 경로가 없다. 남은 query 문자열 생성은 실제 검색을 위한 것이다. 따라서 pointer를 붙여 '현재 도구명/ID decode를 없앴다'고 주장할 수 없다. pointer가 query까지 없애려면 검색 후보집합과 행동 의미가 바뀌며 별도의 방법 실험이 된다.

## 2. 왜 projection 제거가 같은 실험인가

설치된 `/usr/local/lib/python3.13/dist-packages/transformers/models/gemma4/modeling_gemma4.py`를 확인했다. `Gemma4ForConditionalGeneration.forward`의 2574–2602행은 다음 순서다.

```text
outputs = self.model(all multimodal inputs, masks, positions, cache options, ...)
h = outputs.last_hidden_state
logits = self.lm_head(h[:, selected_positions, :])
if final_logit_softcapping is not None:
    logits = cap * tanh(logits / cap)
```

`Gemma4TextModel`은 마지막 `self.norm` 이후의 hidden을 반환한다. 현행 v1.1은 그 norm hook으로 h를 저장하지만, wrapper가 이어 계산하는 마지막 위치의 전체 vocabulary logits도 생성한다. `logits_to_keep=1`은 **한 token position**을 뜻하며 vocabulary를 세 개로 줄이지 않는다.

따라서 learned head의 readout에서는 같은 processor 출력·mask·position·image 입력·dtype·eval mode·cache 설정으로 `self.model.model(...)`을 호출하고 `last_hidden_state[:, -1, :]`를 읽으면 된다. 이는 language-only 모델을 호출해 vision 부분을 생략하는 변경이 아니다. **멀티모달 backbone, vision encoding, 모든 transformer layer는 그대로 실행한다.** 기존 feature가 norm 이후인지 실제 결과로 대조한다.

첫 실험에서는 hook 복사·CPU 전송·`use_cache=True` 등 다른 조건을 함께 최적화하지 않는 편이 해석하기 쉽다. 추후 불필요한 KV 생성 또는 hook 복사를 줄이려면 별도 비용 항과 동치 검증을 추가한다. 기존 action probabilities를 반환하는 encode API를 learned head 전용 hidden API로 나눌 수 있으나, 원래 probability consumer가 있는 경로를 조용히 깨뜨리면 안 된다.

### first-token도 같은 최적화를 받아야 한다

기존 constrained readout은 `softmax(logits[A,B,C])`다. 다음 연산은 수학적으로 같은 분포를 준다.

```text
selected_logits = linear(h_last, lm_head.weight[label_ids], selected_bias_if_any)
selected_logits = original_softcap(selected_logits)
probs = softmax(selected_logits.float(), dim=-1)
```

원래 구현과 같은 dtype·softcap 순서·bias 유무를 유지한다. selected LM rows는 초기화 때 고정 snapshot에서 준비하여 매 요청 큰 weight tensor를 이동하거나 slicing overhead를 숨기지 않는다. 가중치 tying 또는 model reload 시 stale copy를 남기지 않는다.

이는 **A/B/C 조건부 분포**의 동치성이다. 전체 vocabulary 확률이나 '정답 확률'을 계산한 것이 아니다. UPR의 `log P(question | passage)`는 각 question token에서 전체 vocabulary의 정규화 분모가 필요하므로 이 세 행 trick을 그대로 적용할 수 없다. 생성형 query/answer 및 unconstrained generated-action baseline은 실제 vocabulary 탐색이 필요하므로 원래 full head를 유지한다.

수학적 동치는 finite-precision의 bitwise 동치를 보장하지 않는다. BF16에서 full GEMM과 3-row GEMM은 kernel/rounding이 달라질 수 있고 near-tie argmax가 바뀔 수 있다. 결과를 보기 전에 허용 오차와 행동 불일치 처리 정책을 고정한다. 불일치를 '거의 같다'며 무시한 정확도 보존 주장은 하지 않는다.

## 3. 적용 전 통과해야 할 최소 검사

CPU 검사와 실제 GPU 검사를 모두 수행한다. mock CPU 검사만으로 multimodal 동치성을 인증하지 않는다.

1. **CPU 기능 검사:** 선택된 row+bias와 full projection에서 해당 columns를 뽑은 결과, softcap·float32 conditional softmax, legal action mask 및 contextual A/B/C token IDs를 비교한다. learned-only 경로에서 lm_head 호출이 발생하지 않는지 확인한다.
2. **GPU prefix 동치:** 같은 이미 준비한 processor inputs를 재사용하여 원래 wrapper norm-hook h와 backbone-only 마지막 h를 비교한다. 질문+공개 표, text e, image e, 서로 다른 context 길이의 사전 고정 train 사례를 포함한다. `max_abs`, relative error, dtype/shape, 입력 hash를 저장한다. 이미지를 포함한 kwargs/mask/position가 빠지면 실패다.
3. **실제 선택 동치:** 고정된 learned checkpoints의 Q값/argmax와 full-vocab/3-row first-token의 logits·probabilities·argmax를 대조한다. near-tie margin도 남긴다. 순수 동치 arm 채택 조건은 검증 사례의 행동 불일치 0이다. 추후 dev에서도 차이가 발견되면 표기하고 원인 해결 전 속도만 개선된 동치 경로로 해석하지 않는다.
4. **계측:** 동일 warmup, 독점 GPU, 동기화, 같은 input 준비와 CPU transfer 범위로 원래/최적화 경로 순서를 교대하거나 고정 seed로 섞는다. preprocessing·backbone/readout·head·총 decision wall과 peak memory를 분리한다. device-to-host 복사와 작은 CPU MLP 비용도 포함한다.
5. **정책 실행:** 최종 query와 answer는 실제 같은 kernel을 실행한다. 전체 wall에서 절감이 있는지 확인한다. decision 미세 벤치마크 결과를 end-to-end 개선으로 대체하지 않는다.

동일 h가 확인되면 기존 cached training features와 Q-return/aux labels를 재사용할 수 있다. feature prefix 또는 hidden의 의미가 달라지면 재사용을 중단하고 원래 feature를 다시 추출·학습해야 한다. 학습 당시 썼던 비용 label은 acquisition/answer의 미래 비용이며, 이번처럼 선택 후 kernel을 유지하는 변경으로 그 label을 candidate utility처럼 바꾸지 않는다.

## 4. 추가 arm과 비교를 고정하는 방법

진행 중인 v1.1 수집·원래 결과는 보존한다. 최적화용 amendment와 별도 backend/run hash를 남긴다. 원래 labels를 만드는 도중 생성 경로나 입력을 바꾸지 않는다. 새로운 dev 결과를 보고 좋은 head/seed만 골라 속도를 비교하지 않는다.

필수 비교는 다음 네 가지다.

- 기존 full-vocabulary constrained first-token.
- 최적화한 3-row constrained first-token.
- 기존 full-vocabulary prefill + learned head.
- backbone-only prefill + **동일 checkpoint** learned head.

현재 generated baseline은 긴 tool 설명이 아니라 A/B/C를 생성한다. 이를 강한 짧은 생성 baseline으로 유지하며, 존재하지 않는 긴 JSON planner를 현재 baseline이라고 소개하지 않는다. head+aux와 return-only 모두 최적화 혜택을 받아야 한다. 순수 시스템 효과와 auxiliary 학습 효과를 분리한다.

추가 전체 평가 비용을 제한해야 한다면 미리 고정한 lambda와 seed에서 beta=0/1 두 checkpoint를 함께 비교한다. 선택 규칙은 dev 우승자를 고르는 방식이어서는 안 된다. 기존 12개 checkpoint 모두에 적용 가능하면 동일한 정책으로 보고한다. action이 동치인 반복 timing에서 label이나 답변 품질이 새로 개선될 것으로 기대하지 않는다. 동일 action에도 실행된 최종 답이 달라지면 pipeline 비결정성부터 검사한다.

주 지표는 paired question 단위 end-to-end wall 차이와 EM/F1 유지 여부다. decision-only 평균/p50/p95, query generation·retrieval, final answer, 첫 acquisition 및 CPU head 비용을 함께 기록한다. 작은 표본의 p95는 불안정하다고 표시한다. 모델/kernel 준비와 selected-row cache 초기화 비용도 배포 범위에 맞게 별도 기록한다.

진행 담당이 전달한 약 94 train states의 평균은 query generation+retrieval 약 0.392초, final answer 약 0.387초다. 이 문서에서 원시 trace를 재집계한 값은 아니다. 이 비용은 vocabulary readout 생략으로 없어지지 않는다. 실제 절감 상한은 **전체 지연 중 decision의 vocabulary projection이 차지하는 부분**이며 full controller prefill 전체가 아니다. 효과가 작거나 없으면 그대로 보고한다.

## 5. UPR/pointer 후속안을 좁히는 조건

[UPR 원 논문](https://aclanthology.org/2022.emnlp-main.249/)의 점수는 passage에 조건화한 질문 likelihood다. 후보마다 질문을 새로 샘플링하지 않아도 candidate-conditioned forward가 필요하다. 온라인 K개 VLM 점수화는 query decode를 없앴다는 이유만으로 저렴해지지 않는다. 기존 문서의 UPR 해석과 offline teacher 제안은 이 점에서 타당하다.

full-state joint pointer와 cached-state joint pointer는 연구할 만하지만 다음 조건의 **새 실험**이다.

1. 관측만으로 만든 동일 shortlist를 모든 pointer/first-token/heuristic baseline에 제공한다. 새 query가 필요한 경우를 위한 fallback의 허용 규칙과 비용을 고정한다.
2. `STOP`, `(TEXT,i)`, `(IMAGE,j)`, 필요 시 `NEW_QUERY/REFRESH`의 downstream return을 같은 단위로 수집한다. 현 Q3의 TEXT/IMAGE return은 특정 생성 query+top1 kernel의 outcome이며 후보 i의 label로 복사할 수 없다.
3. 작은 K에서 candidate coverage와 observed oracle 여지를 먼저 확인한다. shortlist 밖 정답을 pointer가 찾을 수 없다는 한계를 보고한다. gold support는 평가에만 쓴다.
4. 동일 capacity·label·수집 비용의 outcome-only joint head, first-token candidate pointer, fixed ranking과 비교한다. UPR score는 return과 다른 relevance auxiliary이며 offline teacher로 쓸 경우 수집 GPU 비용도 맞춘다.
5. image candidate embedding/metadata만 보는 cached pointer를 full visual understanding이라고 부르지 않는다. 전체 후보 이미지를 VLM에 넣으면 이미 후보 visual-read 비용을 썼다는 점을 계측한다.

기존 성공적인 v1.1 head에서 바로 UPR 또는 pointer로 온라인 경로를 교체하는 것은 승인하지 않는다. 동등 kernel의 projection 최적화를 먼저 끝내고, candidate labels와 고정 shortlist의 작은 탐색 pilot 하나만 후속으로 선택한다. cached representation, refresh, UPR, early exit를 동시에 붙이지 않는다.

## 6. KV 재사용의 경계와 주장

현재 `DECISION`, query instruction, answer instruction이 다르므로 마지막 hidden 및 **전체 prompt KV**를 서로 대체할 수 없다. 같은 question/evidence 부분이 실제 token prefix로 일치하는지, 마지막 multimodal attention block이 닫혀 있는지, position/cache semantics가 같은지 검증한 뒤 공통 prefix만 공유할 수 있다. 재사용하는 것은 task-specific 마지막 hidden이 아니라 검증된 prefix cache다. 새 prompt layout을 만들었다면 trained feature가 바뀌므로 재추출 여부를 판단해야 한다.

이번 projection 최적화는 새 학습 알고리즘도, UPR의 새로운 변형도, 처음 제안한 직접 도구 호출 구조도 아니다. **동일 모델·정책의 불필요한 계산을 제거한 시스템 최적화**로 정확하게 보고한다. 기존 연구 질문인 접근 대조 supervision의 추가 효용은 최적화된 first-token과 generic value baseline에서도 별도로 살아남아야 한다.
