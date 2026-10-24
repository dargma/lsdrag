# 연구 주장: 하나의 reader backbone에서 장기 의사결정과 생성 통합

## 사용자 의도와 주장 범위

사용자는 단순 tool classifier가 아니라 미래 turn들의 최종 정확도와 비용을 고려하는 decision policy를 구현하고 싶다고 명확히 했다. 별도 reader LLM과 decision LLM을 배치하기보다, 같은 reader backbone에 작은 policy head를 붙여 선택과 생성을 함께 수행하고 A-RAG 효율을 개선하는 것이 논문 목표다.

핵심 주장의 후보 문장: **공유 reader 표현 위에 장기 결과로 학습한 경량 행동 정책을 결합하여, 별도 의사결정 LLM 없이 검색·읽기·종료를 제어하고 같은 backbone으로 인자와 답변을 생성한다.** 성능/효율 부분은 독립 평가로 입증되기 전까지 연구 가설이다.

'동시에 생성'은 실행 의미를 정확히 써야 한다. 현재 구현은 공유 prefill에서 행동을 선택하고 그 KV cache를 재사용해 선택한 행동의 인자/답변을 autoregressive 생성한다. 모든 선택과 완성된 답변을 한 번의 forward로 동시에 산출하는 것은 아니다. 하나의 backbone/공유 계산으로 선택과 조건부 생성을 통합했다는 표현이 적절하다. 현재 native 대비 decoder 경로 차이와 forced-prefix 토큰을 공개하며 latency 우월성은 통제 실험 전 주장하지 않는다.

## 정책 여부는 크기보다 목적과 검증의 문제

이력 h_t를 동결 reader가 표현 z_t=f_θ(h_t)로 바꾸고, head는 π_ϕ(a_t|z_t)를 출력한다. 같은 reader가 p_θ(y_t|h_t,a_t)로 인자 또는 최종 답변을 생성한다. 결합 정책은 다음처럼 표현할 수 있다.

$$p_{\theta,\phi}(a_t,y_t\mid h_t)=\pi_\phi(a_t\mid f_\theta(h_t))\,p_\theta(y_t\mid h_t,a_t).$$

θ를 동결해도 ϕ가 최종 episode 성과로 학습된다면 정책 학습은 가능하다. 반대로 큰 별도 모델도 현재 행동 라벨 모방만 했다면 장기 목적을 최적화했다는 근거가 없다. 'policy'와 명시적 미래 시뮬레이션을 수행하는 'planner/world model'도 구분한다. 기대 장기 성과를 학습한 reactive policy는 추론 시 미래를 명시적으로 rollout하지 않아도 된다.

연구 목적의 수식 예시는 다음과 같다. **이것은 H3가 직접 최적화한 손실이 아니라 앞으로 검증하려는 episode 수준 목적이다.**

$$\max_\phi\;\mathbb{E}_{\tau\sim\pi_\phi,p_\theta}[F1(\mathrm{answer}_T)]\quad\text{s.t.}\quad\mathbb{E}_{\tau}[\sum_t C_t]\le B.$$

또는 학습용 scalarization:

$$J(\phi)=\mathbb{E}_{\tau}[F1_T-\lambda\sum_t C_t],$$
$$Q^{\pi_k}(h_t,a)=\mathbb{E}[F1_T-\lambda\sum_{u=t}^{T}C_u\mid h_t,a_t=a,\pi_k].$$

어느 λ/가중치도 dev 결과에 맞춰 반복 조절하지 않는다. 목표 판정은 기존 고정 기본 GRPO 대비 F1+5%p/생성≤70%의 동시 달성과 유용한 상태별 행동 근거로 유지한다. 단순 scalar reward 상승으로 목표를 대체하지 않는다.

## H3가 이미 한 것과 아직 하지 않은 것

H3는 행동명 정답만 모방한 것이 아니다. train 상태에서 지금 종료/계속을 개입하고 끝까지 답변을 생성한 F1과 남은 생성 비용을 타깃에 사용했다. 즉 학습 신호는 delayed outcome을 포함한다.

하지만 suffix는 고정 H1정책이다. H3 자체가 바꾼 방문 상태와 후속 결정에 대한 Q^H3를 반복 갱신하지 않았다. 현재는 H1의 downstream outcome에 대한 제한된 종료 정책 개선에 가깝다. 또한129개 파라미터 보정은 FINAL만 바꾸며 검색/읽기 사이의 순위를 바꾸지 않는다. 이 한 실험으로 전체 장기 retrieval policy를 학습했다고 말할 수 없다.

따라서 가장 먼저 보완할 축은 head 크기나 backbone unfreezing이 아니라 **현재 정책의 경로에서 실제 장기 결과를 재수집하고, 종료뿐 아니라4개 행동에 대한 개선을 다시 학습하는 policy iteration**이다. 이것은 PPO/GRPO 도입을 필수로 뜻하지 않는다. 기존 offline outcome-supervised policy improvement를 반복하는 bounded 방식으로 시작할 수 있다. 과거 PPO/GRPO 실패와 native계약/사용자선호는 보존한다.

정책 개선 반복의 제안:

1. 현재 후보를 고정해 train 문항에서 rollout한다.
2. 그 후보가 실제 방문한 상태에서 행동 개입을 비교하고 같은 후보로 끝까지 진행한다.
3. 최종 품질/비용에 근거해4행동 head를 한 번 개선한다. 필요한 탐색의 지연된 이익과 불필요한 실행을 함께 반영한다.
4. 동일 개발 문항의 실제 QA로 후보 하나를 판정한다. 독립 확인용 reserved80은 이후 별도 동결 전 접근하지 않는다.

구체 split/보상/학습 설정은 아직 동결하지 않았고 이 문서 자체로 새 실행을 시작하지 않았다. 현재 승인된 H3 상수 대조 실험을 완료한 후 한 방향만 선택한다. 무제한 분석/구조 비교로 확장하지 않는다.

## Frozen backbone으로 충분한가?

가능성과 충분성은 다르다. 기존 표현에 필요한 이력/증거 충분성이 담겨 있고 작은 head가 읽어낼 수 있다면 head-only policy learning은 가능하다. 그러나 현재3개 요약 hidden과 frozen128차원 H1표현이 검색 결과 간 관계, 아직 부족한 근거, 이전 실패를 구분하기에 충분한지는 미입증이다.

초기에는 frozen을 유지한다. 정책의 현재 방문 상태와 장기 타깃을 맞추는 개선을 먼저 검증한다. 충분한 학습 근거와 타깃 반영에도 새 상태에서 판단이 반복 실패할 때, 관측 표현을 직접 읽는 head 또는 작은 adapter가 필요한지 후보 하나로 검증한다. 실패를 곧바로 'frozen LLM으로는 불가능'으로 해석하지 않는다. 반대로 frozen/head-only를 논문 문구를 위해 절대 조건으로 고집하지도 않는다. 작은 adapter가 필요해도 단일 backbone의 선택·생성 통합 주장은 유지될 수 있다. 현재 unfreeze/adapter실행은 제안일 뿐 미승인 신규 프로토콜이다.

## 선행연구와 신규성의 경계

- [LLaRP, ICLR2024](https://llm-rl.github.io/), [논문](https://arxiv.org/abs/2310.17722): 동결 LLM과 학습되는 관측 인코더/행동 디코더를 사용한 embodied RL 정책. frozen LLM 기반 정책의 선례이며 head만 붙이는 일반적 발상 자체가 새롭다는 주장을 제한한다. 우리 RAG 및 같은 LLM의 언어 생성 재사용 효용을 입증해 주는 자료는 아니다.
- [HELM, ICML2022](https://proceedings.mlr.press/v162/paischer22a.html): 동결 pretrained language Transformer로 이력을 표현하고 actor-critic에 사용하는 선례. 표현 동결과 정책 학습이 양립할 수 있다는 근거이며 우리 head 충분성의 증명은 아니다.
- [Agent Lightning](https://arxiv.org/abs/2508.03680): agent execution을 MDP로 다루고 trajectory를 training transitions로 연결하는 credit assignment 접근. 장기 성과를 행동에 연결하는 문제의 관련 문헌이며 해당 프레임워크 도입/재현을 의미하지 않는다.

우리 연구의 입증 대상은 (1)별도 decision LLM 없는 공유 계산, (2)최종 결과로 학습한 상태별 장기 계산 배분, (3)독립 QA의 품질/비용 동시 개선이다. 현재 H3는 유망한 비용 중심 후보이며 세 가지 모두를 입증한 완성 결과는 아니다.
