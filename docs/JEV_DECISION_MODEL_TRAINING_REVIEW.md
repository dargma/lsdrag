# Decision 모델 최신 학습법 확인 — Jev·Laya·Kev·SemIf

2026-10-14 확인. **실험에는 외부 decision API를 사용하지 않고 현재 로컬 Gemma 4 E4B-it에 head를 붙이거나 미세조정한다.** 아래는 그 설계를 위한 공식 문서·공개 학습 코드 조사다.

**핵심 결론:** ‘decision 모델이면 반드시 RL’은 아니다. 공개 사례에는 지도학습으로 head·LoRA를 맞추는 방식, proper-scoring reward를 더하는 방식, 학습 없이 기존 LM의 선택지 logits를 읽는 방식이 모두 있다. 먼저 작은 head의 cross-entropy 기준선을 검증하고, 라벨·확률 보정·도메인 전이를 확인한 뒤 복잡도를 늘리는 편이 타당하다.

## 1. 무엇이 공개되어 있는가?

| 모델·구현 | 입력→결정 경로 | 확인한 학습 방법 | 우리 설계에 참고할 점 |
|---|---|---|---|
| **Jev / TypeSafe** | state + typed questions → bounded decision·분포 | 공식 명칭은 **RLCD**. 구체 loss·데이터·가중치·전체 구조는 확인 범위에서 비공개 | interface·확신도 개념 참고. 내부 알고리즘 복제라고 주장하지 않음 |
| **Laya** | encoder + option 기반 decision head | soft cross-entropy, 또는 noisy-logit policy-gradient+proper reward를 더하는 독립 `rlcd` 구현 | loss·보정·collapse 감시를 실제 코드로 비교 가능 |
| **Kev-4B** | frozen Qwen backbone + LoRA + pointer head | 선택지 CE, 최소 반사실 쌍·옵션 순서 섞기, 뒤에 temperature calibration | 현재 Gemma의 head→LoRA 경로에 가장 직접적인 공개 학습 사례 |
| **SemIf-OpenJev 변형** | frozen LM의 선택지 logits 직접 읽기 | 기본 readout은 새 학습 없이 사용 | 학습 없는 대조군. logits가 바로 정답 확률은 아님 |

출처: [Jev 공식 소개](https://docs.typesafe.ai/introduction) · [TypeSafe 공식 연구 방향](https://typesafe.ai/) · [Laya 학습 코드](https://github.com/NandhaKishorM/laya/blob/main/laya/train.py) · [Kev-4B 모델 카드](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md) · [SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev).

SemIf라는 이름의 여러 프로젝트·포크가 있으므로 서로 다른 구현의 속도·모델·학습법을 하나의 결과로 섞지 않는다. 위 표는 연결한 구현을 기준으로 한다.

## 2. Jev — 공식 RLCD와 비공개 부분을 구분

TypeSafe 공식 사이트는 RLCD를 **Reinforcement Learning for Calibrated Decisions**로 소개하며 새로운 architecture·sampler·training algorithm을 사용한다고 설명한다. 공식 API 문서는 Choice·Score·Noul과 분포·confidence를 설명한다.

그러나 이 공식 설명만으로 아래를 확정할 수 없다.

- backbone 모델·층수·attention·decision head의 구체 구조.
- reward·policy-gradient·calibration의 정확한 수식과 데이터 구성.
- weights·optimizer·sampling·학습량과 재현 가능한 학습 코드.

따라서 Kev·Laya의 독립 재구성이나 서드파티 reverse engineering을 Jev의 공개된 실제 구현으로 소개하면 안 된다. 제한된 선택지만 출력하더라도 **잘못된 선택은 가능**하다. 선택지 밖 문장을 생성하지 않는다는 사실이 의미적 정답을 보장하지 않는다. [공식 Jev 문서](https://docs.typesafe.ai/introduction) · [공식 홈페이지](https://typesafe.ai/)

## 3. Kev — 지도학습 + LoRA + pointer head

확인 대상은 Kev-4B 모델 카드의 Kev 1.0, 2026-09-24 공개 체크포인트다. frozen Qwen3.5-4B-Base에 **LoRA rank 16, α 32**와 pointer head를 붙인다. 우리 실험에서 이 Qwen 모델로 Reader를 교체한다는 뜻은 아니다.

### head가 무엇을 학습하는가?

질문의 마지막 토큰 표현과 각 선택지 경계 토큰 표현을 투영해 점수를 만들고 softmax를 적용한다. fixed 4-class linear head와 달리, 입력에 주어진 선택지 각각을 점수화하는 pointer 방식이다. context를 한 번 계산하고 독립 질문 분기에서 재사용하는 구현도 제공한다.

학습 손실은 선택지의 **cross-entropy**다. 옵션 순서를 섞고, distractor·none-of-the-above를 넣으며, 정답이 있는 입력과 정답을 제거한 입력의 최소 쌍을 만든다. 이후 날짜·근거 부족·긴 문서·도구 과제 등을 학습하면서 기존 데이터를 replay한다. Jev 출력으로 학습하지 않았다고 명시한다. [Kev-4B 모델 카드](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)

### 확률 보정은 별도 단계

공개된 Kev-4B는 dev 질문 1,264개에서 NLL을 최소화한 단일 temperature **T=2.41**을 적용한다. 같은 training corpus의 held-out item 기반이며, 모든 새 도메인에 동일한 보정이 유지된다고 가정할 수 없다. 온도는 선택 확률의 뾰족함을 바꾸지만 같은 logits의 argmax를 바꾸지 않는다.

우리 train 성공 문항은 현재 4개뿐이므로 이 사례의 보정 절차를 그대로 적용해 ‘calibrated’라고 부를 수 없다. raw 분포와 실제 오류 관계부터 보고해야 한다.

## 4. Laya — 이름보다 실제 loss를 확인

Laya의 공개 `laya/train.py`는 `soft-ce`와 `rlcd`를 구분한다. 확인한 구현에서 `rlcd`는 logits에 Gaussian noise를 주고, proper reward로 평가한 표본의 정규화 advantage를 Gaussian log-density와 결합한 policy-gradient 항에 **soft cross-entropy를 더한다**.

이것은 Laya의 독립 공개 학습 방법이다. Jev의 비공개 RLCD와 동일한 방법이라고 쓰지 않는다. 무엇보다 현재 코드 머리말은 해당 typed-decisions split에서 **추가 RL 항의 이득을 관측하지 못했다**는 결과를 명시한다. 따라서 우리도 CE 없이 RL만 시작할 이유가 없다. [실제 학습 코드](https://github.com/NandhaKishorM/laya/blob/main/laya/train.py)

공식 README는 typed-decision 파인튜닝, calibration, 언어·라벨 표현·context budget·확신도 한계를 함께 기록한다. 노트북의 calibration sample이 train item에서 왔다는 주의도 있으므로 그대로 독립 검증처럼 복사하면 안 된다. 학습·보정·test를 분리해야 한다. [Laya 공식 저장소](https://github.com/NandhaKishorM/laya#fine-tuning)

## 5. SemIf — 새 head 없이 기존 선택지 logits 읽기

연결한 SemIf-OpenJev 구현은 런타임에 선택지·설명을 넣고 LM의 해당 선택 logits를 한 번의 forward에서 읽는다. 텍스트 답을 샘플링하지 않고 제한된 후보 위 분포를 계산하는 방식이다. 별도 head 학습과는 다른 대조군이다. [공식 upstream](https://github.com/TheoLeeCJ/SemIf-OpenJev)

우리 Gemma에 적용할 때도 아래를 확인해야 한다.

- 선택지 표기가 실제 suffix 문맥에서 단일 토큰인지, 아니면 여러 토큰의 likelihood 평가가 필요한지.
- 추가 선택 안내로 prompt가 바뀌는지, original tool template·이미지 입력이 그대로 보존되는지.
- 선택지로 정규화한 분포와 전체 vocabulary 분포에서 후보가 차지하는 질량을 구분하는지.
- scoring 뒤 native 인자 생성을 위해 다시 prefill해야 하는 비용이 포함되는지.

새로 학습하지 않는다는 것이 빠르거나 정확하다는 보장은 아니다. 동일 Gemma의 실제 상태에서 확인해야 한다.

## 6. 최신 평가에서 얻는 주의점

[Jev 평가 논문, 2026-09-29](https://arxiv.org/abs/2609.37647)은 다수 task의 선택 성능·확률을 측정한다. 특정 언어·라벨·임계값에서의 결과를 A-RAG 성공률로 옮길 수 없다.

[Fast Models, Slow Evidence, 2026-10](https://arxiv.org/abs/2610.02267)은 gate 자체의 점수를 전체 품질로 해석하거나, pre-screen 비용을 빼거나, 평가 자료에서 임계값을 맞추면 결론이 바뀐다고 분석한다. 우리 실험은 head·인자 생성·fallback·retrieval·최종 답변을 모두 포함한 지연과 실제 EM을 함께 보고한다.

## 7. 우리 Gemma 학습에 반영한 결정

| 순서 | 학습·구현 | 이유·전제 |
|---|---|---|
| 1 | 기존 생성 경계의 frozen hidden → linear head, CE | 가장 작은 모델 변경으로 action 분리 가능성 확인 |
| 2 | 성공 궤적의 singleton tool·FINAL 모방 | 현재 라벨의 의미와 지원 범위를 정확히 제한 |
| 3 | train에서 실제 증거 제거·상충 최소 쌍 구축 | 근거 충족·수정 판단을 단순 행동 모방과 구분 |
| 4 | 필요한 경우 Gemma LoRA + head | frozen 표현이 decision에 부족한지 확인한 뒤 수행 |
| 5 | 충분한 별도 dev에서 보정·선택적 fallback | class 확률과 실제 오류 관계 검증 |
| 6 | 학습 예산을 맞춘 RL/proper-reward 대조 | CE로 해결되지 않는 문제가 있을 때만 후속 검토 |

초기 모델은 fixed action set의 head이므로 Kev/Laya의 범용 dynamic-option 모델과 같은 범용성을 주장하지 않는다. 학습 라벨이 없는 class를 다룬 척하지 않으며, 새 학습 자료와 heldout 평가가 선행되어야 한다.

[구체 실험 계획](JEV_EXPERIMENT_PLAN.md) · [기여 가설](JEV_CONTRIBUTIONS.md) · [Astra의 최신 RAG·memory 조사](JEV_MEMORY_RESEARCH_ASTRA.md).
