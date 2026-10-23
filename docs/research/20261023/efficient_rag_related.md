# 효율적 agentic RAG·종료 정책 관련 연구

조사 기준일: 2026-10-23. 원문 HTML의 방법·실험·부록을 확인했다. 아래 날짜는 링크한 arXiv v1 게시일이며, 최신 개정판 전체를 추적했다는 뜻은 아니다. 검색 범위에는 2026년 9–10월도 포함했으나, 이 노트에서 확인한 가장 최근 직접 관련 원문은 8월25일 MetaRAG다. 결과 수치는 저자 보고이며 독립 재현하지 않았다. 기존 연구 메모의 링크는 탐색 출발점으로만 사용했다.

우리 비교 대상은 frozen Qwen3-4B-Instruct-2507의 hidden state에 작은 head를 붙이고, keyword/semantic/read/FINAL을 선택하며 같은 LLM이 인자·답변을 생성하는 pilot2다. 공통 reference 정책의 suffix rollout으로 얻은 품질·남은 비용을 학습한다. 이는 PPO가 아니라 all-action Monte Carlo policy-improvement surrogate다.

## 원문에서 확인한 핵심 선행

| 연구 / 최초 게시 | 방법·학습 데이터·학습 대상 | 실행·학습 예산과 우리 방법의 차이 |
|---|---|---|
| [Search-R1](https://arxiv.org/html/2503.09516v1), 2025-03-12 | NQ+HotpotQA에서 최종 EM 보상으로 검색과 생성 LLM을 PPO/GRPO 학습. 검색 관측 토큰은 loss에서 제외. | Qwen2.5 3B/7B, Llama3.2 3B; E5/Wiki2018 top3. 종료는 answer 생성 또는 검색 상한. 본문에서 wall-clock/GPU 총비용은 확인 못함. 우리처럼 head만 갱신하지 않는다. |
| [Stop-RAG](https://arxiv.org/html/2510.14337v1), 2025-10-16 | 항상 끝까지 수행한 trajectory의 각 prefix에서 8회 답변을 생성해 F1을 추정. full-width Q(λ) target으로 별도 DeBERTa-v3-large와 STOP/CONT 두 head 학습. | MuSiQue/HotpotQA/2Wiki train, 각 dev에서 val1000/test1000; 최대10회. validation에서 checkpoint·Q-margin 임계값 선택. 미래 계속검색 가치 학습은 이미 선행. 우리는 독립 encoder를 추가하지 않고 4개 행동을 통합한다. |
| [Search-R1++](https://arxiv.org/html/2602.19526v1), 2026-02-23 | Fast Thinking으로 검색/답변을 직접 생성. F1 reward의 answer-avoidance 붕괴와 action penalty, PPO/GRPO/REINFORCE를 비교. NQ+HotpotQA에서 policy LLM을 업데이트. | 8×A100, 600steps, global batch512, context4096. 작은 head 연구는 아니다. “긴 thinking 없이 정책 학습” 자체는 차별점이 약하며, 알고리즘 순위는 해당 설정의 결과이지 보편 명제가 아니다. |
| [GRASP](https://arxiv.org/html/2607.10463v1), 2026-07-11 | keyword/semantic/read/answer 행동으로 Qwen2.5-3B-Instruct 전체 파라미터를 GRPO 학습. 답변F1+gold reading+상보 검색+정답 조건부 turn 효율 보상. | HotpotQA 17,280 prompts, 270steps, 4rollouts/prompt, 최대10turn/6144tokens. 2×A100 80GB와 별도 검색 GPU. 행동 공간이 거의 동일하므로 도구 구성 자체의 novelty는 주장할 수 없다. |
| [HALT](https://arxiv.org/html/2608.02009v1), 2026-08-03 | host agent를 동결하고 필요한 hop claim들이 근거로 모두 뒷받침될 때 종료. 별도 Qwen2.5-3B LoRA+3-way verifier; HotpotQA 4,903질문/9,626rows, 1000steps. | Self-Ask 3B/7B, 최대8loops. gold claim은 진단 상한, 질문에서 생성한 claim이 배포 가능한 비교군. 검색 감소에도 verifier 때문에 raw token 수가 늘어남을 명시한다. 우리는 동일 prefill 재사용으로 이 추가 모델 비용을 피하려는 설계다. |
| [When Should Multi-Round RAG Stop?](https://arxiv.org/html/2608.13237v1), 2026-08-13 | frozen Search-R1 위에 별도 Qwen3.5-2B structured judge LoRA. HotpotQA 900질문/3,009states; 3epochs, 12,654steps. sufficiency/gap을 생성하고 grouped validation에서 모델·임계값을 고정. | confirmatory800에서 검색3.70% 감소, EM−0.625pp; early stop69개 중 unsafe27개. 총 추론비용 감소는 미입증이라고 명시. state 분류 점수와 실제 최초 STOP 정책의 성능을 분리해야 한다는 직접적 근거다. |
| [MetaRAG](https://arxiv.org/html/2608.24214v1), 2026-08-25 | Qwen2.5 3B/7B에 verify-first 행동 생성을 학습. 학습 때만 같은 policy의 추가 yes/no forward로 answerability belief를 추정하고, 최종 EM으로 gated된 행동-믿음 일치 보상 사용. NQ+HotpotQA, GRPO. | 8×A100, group5, 최대4turn. 별도 judge 없이 outcome-aware decision supervision이 가능하다는 선행. 다만 rollout 반사실 Q가 아니라 자기 belief이고, 추론 중 verification rationale은 유지된다. 작은 frozen-backbone head는 아니다. |

원문 검토 위치: Search-R1 §3–4; Stop-RAG §4–5; Search-R1++ §3–5/App.A.3; GRASP §3/App.C; HALT §3.3/§5.7/App.E; structured stopping §4/§6; MetaRAG §3/§4.1. 방법을 확인한 수준이며 소스코드 실행·데이터 재현·통계 재계산은 수행하지 않았다. 공개 repository를 갖춘 논문도 구현과 본문의 완전한 일치를 검증한 것은 아니다.

## 우리 주장에 미치는 영향 — 연구자의 해석

**“미래 결과를 보고 종료를 배운다”, “검색·읽기를 RL로 선택한다”, “독립 judge 없이 학습한다”만으로는 contribution이 부족하다.** 각각에 해당하는 선행이 존재한다. 우리 후보 주장은 다음의 결합이 실제 품질–비용 이득을 만드는지다.

> 생성 backbone을 동결한 채 이미 계산한 hidden state에서 작은 action head만 학습하여, 인자·답변 생성 능력을 유지하면서 도구 구문 생성과 불필요한 탐색을 함께 줄인다.

구문을 생략하는 runtime 이득과 더 좋은 행동을 학습한 policy 이득은 서로 다른 결과다. 기존 모방 head 대비 행동 변화와 QA/비용 변화가 없으면 후자는 성립하지 않는다. 동일 feature 차원이나 head 크기가 작다는 사실만으로 효율·일반화 우월성을 주장할 수도 없다.

## 후속 비교 실험에 필요한 기준 — 제안이며 현재 동결 실험에 사후 추가하지 않음

1. **동일 생성 조건의 Native**: thinking OFF, 같은 모델·corpus·parser·인자 생성·최대turn. 도구명/고정 prefix를 강제로 공급하는 최적화 Native를 별도 두어 head의 정책 효과를 분리한다.
2. **정책 학습 대조**: 같은 state/action/rollout 데이터를 쓰는 BC, outcome-CE, return-based head를 비교한다. 동일 데이터뿐 아니라 collector의 reference 정책 편향도 명시한다.
3. **종료만 학습하는 ablation**: 같은 hidden state에서 STOP/CONT만 학습하고 나머지 행동은 고정해, 전체4-action 개선이 단순 조기종료 이상인지 판별한다. Stop-RAG식 target을 적용한다면 별도 방법으로 이름 붙이고 충실 재현과 구분한다.
4. **저렴한 규칙 대조**: 고정1/2/3-step, 중복검색 중단, validation에서 고정한 stopping threshold. state accuracy보다 전체 trajectory의 첫 종료 위치와 잘못된 조기종료를 평가한다.
5. **비용 분해**: 검색호출, read호출, 생성token, 입력/prefix token, 추가 decision/verifier forward, VRAM과 학습·수집 총비용을 각각 기록한다. 검색 수 감소를 총비용·시간 감소로 치환하지 않는다.
6. **성공 기준**: 품질–비용 곡선과 paired CI, cap-forced FINAL, 과소검색/과다검색, 실패 포함 전체질문 결과를 보고한다. 수집예산·데이터노출·단일seed 한계를 보존하고 noninferiority가 필요하면 허용차를 평가 전에 고정한다.

현재 pilot2의 head softmax는 행동 선택 분포다. 위 논문들의 Q-value, answerability belief, evidence sufficiency와 의미가 다르다. 이를 “정답 확률”로 해석하려면 별도 outcome supervision·calibration·held-out 검증이 필요하다.

## 논문에서 가져올 구체 방법과 채택 조건

우선순위는 Stop-RAG에서 가져온 **종료만 학습하는 작은 대조군**이다. 기존 train 문항의 정책 trajectory를 최대예산까지 수집하고, 각 prefix에서 STOP 답변과 CONT 이후 답변의 품질·남은 비용을 얻는다. 같은 frozen LLM hidden feature로 두 출력 head를 학습하면 전체4-action policy가 얻는 이득 중 얼마가 종료 개선인지 분리할 수 있다. 원논문의 별도 DeBERTa, 8회 답변샘플, Q(λ)·annealing·validation threshold까지 맞추지 않는 한 “Stop-RAG 재현”이라 부르지 않고 **Stop-RAG-inspired shared-feature adaptation**으로 보고한다. 짧은 pilot에서는 동일 branch 샘플을 재사용해 예산을 맞추되, 그것이 원래 full-horizon 데이터와 다름을 기록한다.

HALT에서 가져올 것은 별도3B verifier 설치보다 **지표 분해와 evidence 기반 failure analysis**다. 정답 유지 여부와 종료 시점에 필요한 evidence가 확보됐는지는 서로 다르므로 train/validation의 gold support는 진단에만 사용한다. 배포 head의 입력에는 넣지 않는다. 문헌의 gold-claim 성능을 배포 성능으로 인용하지 않는다.

Structured stopping 연구에서 가져올 것은 **question 단위 split, 첫 STOP 정책 평가, 임계값 동결**이다. train state를 여러 개 만든 뒤 state 단위로 무작위 분할하면 같은 질문의 prefix가 새어 들어갈 수 있다. 따라서 question/family를 먼저 분할한다. 기존 exposed test100·pilot1 test20과 현재 pilot2 test40은 후속 튜닝·threshold 선택에서 제외한다. 새 후속 실험은 train/validation 안에서 방법을 정하고 독립 fresh test를 outcomes 전에 동결한다.

채택 gate는 미리 정한다: (a) 데이터·샘플·생성 경로를 맞춘 BC/CE/규칙 대비 품질 손실 허용차 이내, (b) tool calls뿐 아니라 전체 생성/입력 token 비용 감소, (c) cap 종료나 오류 증가로 만든 가짜 절감이 없음, (d) 동결 test의 paired CI와 여러 seed에서 방향 일관성. 구체 허용차·sample 수는 새 protocol에서 고정해야 하며 현재 결과를 보고 거꾸로 고르지 않는다. 시간 우월성은 별도의 통제 측정 전에는 주장하지 않는다.
