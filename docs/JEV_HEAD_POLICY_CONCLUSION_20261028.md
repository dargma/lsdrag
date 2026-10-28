# 결론 보고: selection head는 도구 선택은 되지만 정확도·토큰 절감 정책은 안 된다 (2026-10-28)

## 요약

- head는 **도구 선택**은 잘한다. 고른 행동이 대안보다 낫고, 형식 오류도 없다.
- head를 **정책**으로 GRPO 학습해도 정확도는 오르지 않았다(4B, 9B 모두).
- 관측된 **토큰 절감은 대부분 실패 사례**(둘 다 오답, 또는 정답을 잃은 문항)에서 나왔다. 공통 정답 문항에서는 거의 줄지 않았다.
- 정책이 배운 것은 "정답을 더 효율적으로 찾기"가 아니라 **"가망 없는 문항을 빨리 포기하기"**에 가깝다.

모든 평가는 반복 사용한 개발 문항에서 했다. 독립 일반화 근거는 아니다.

## 1. 실험 구성

### 1.1 에이전트

- **도구:** keyword_search, semantic_search, read_chunk, FINAL(4개 행동)
- **코퍼스:** MuSiQue train+dev 전체 문단을 중복 제거한 공유 코퍼스 `musique_shared_20261023`
- **종료 규칙:** 도구 호출 8회 이후에는 FINAL을 강제한다.

**매 결정 단계의 흐름**
1. LLM이 현재 컨텍스트를 prefill한다.
2. head가 그 hidden으로 행동을 고른다.
3. LLM이 그 행동의 인자(또는 답 JSON)를 문법 제약 하에 생성한다.

**LLM은 학습하지 않는다.** 아래 모든 실험은 head만 학습했다.

### 1.2 head 구조

| | 9B head (`StateHead`, value_aux) | 4B head (`PolicyHead`) |
|---|---|---|
| LLM | Qwen3.5-9B (hidden 4096) | Qwen3-4B-Instruct-2507 (hidden 2560) |
| 입력 위치 | 마지막 층 hidden의 3곳: 마지막 토큰, 질문 끝 토큰, 최근 관측 구간 평균 | 마지막 층 hidden의 마지막 토큰 1곳 |
| 입력 차원 | 3 × 4096 = 12,288 (학습 데이터 평균·표준편차로 정규화) | 2,560 (정규화) |
| 본체 | Linear(12288→128) → Tanh → Linear(128→4) | trunk Linear(2560→128) → Tanh → action Linear(128→4), skip Linear(2560→4) |
| 파라미터 수 | 약 157만 | 약 34만 |
| 출력 | 4개 행동 logit. 불가능한 행동(관측 전 read_chunk 등)은 −10⁹으로 마스크 | 같음 |
| GRPO 학습 대상 | MLP 전체(정규화 통계 고정) | action, skip만 (trunk 고정) |

### 1.3 데이터셋

모든 문항은 MuSiQue 2hop이다. 학습과 평가 문항은 서로 겹치지 않는다.

| 용도 | 문항 | 쓰인 곳 |
|---|---|---|
| head 사전학습(모방) | train192: native 9B가 고른 행동 | 9B bootstrap head |
| head 결과 학습 | train48(train192의 부분집합): 상태마다 네 행동을 실행해 본 분기 결과 | 9B value_aux head |
| GRPO 학습 | 9B: train48 × 4궤적 = 192궤적, 결정 880개 | 9B GRPO48, 토큰 벌점 비교 |
| | 4B: train192 × 4궤적, 2회 반복 | 4B 공유 코퍼스 GRPO |
| | 4B: train96 × 4궤적, 3회 반복 = 1,152궤적 | 4B GRPO96 |
| 평가 | dev64: 64문항, 학습 전후를 같은 문항으로 비교 | 9B 전 실험, 4B 공유 코퍼스 |
| | 새 80문항(decomposition family 비중복) | 4B GRPO96 |
| 진단 | train48 에이전트 궤적, dev64 | 분기 분석, 천장 실험, ④ 선별 |

- train192와 dev64는 4B와 9B가 같은 split(`decision_state_policy_iteration_20261023`)을 쓴다.
- GRPO96은 공유 코퍼스 이전 설정이라 다른 결과와 직접 비교하지 않는다.

### 1.4 학습 objective

#### (1) 9B head 사전학습

**bootstrap (모방):** native 9B 에이전트가 고른 행동 aᵗ를 따라 하도록 학습한다.

$$\mathcal{L}_{BC} = \sum_s w_s\big[\,0.95\,(-\log \pi_\theta(a^t_s|s)) + 0.05\,(-\tfrac{1}{|\mathcal{A}_s|}\textstyle\sum_{a\in\mathcal{A}_s}\log\pi_\theta(a|s))\big]$$

- 두 번째 항은 가능한 행동 𝒜ₛ에 대한 label smoothing이다.
- wₛ는 문항별로 합이 같도록 맞춘 가중치다. 상태가 많은 문항이 학습을 지배하지 않게 한다.
- 설정: AdamW 3e-4, 100 step.

**value_aux (결과 기반 보정):** 각 상태에서 네 행동을 모두 실행해 본 결과(F1, 생성 토큰)로 목표 행동 a*를 정한다.
- a*의 조건: 정답성(EM, F1)이 현재 행동 이상이고, F1이 더 높거나 토큰이 더 적다.

$$\mathcal{L} = \tfrac12\sum_{s\in C} w_s\,\mathrm{CE}\big(\tfrac12\pi_0(\cdot|s)+\tfrac12\,\mathbf{1}_{a^*_s},\ \pi_\theta\big) + \tfrac12\sum_{s\in R} w_s\,\mathrm{CE}\big(\pi_0(\cdot|s),\ \pi_\theta\big) + 0.1\,\mathcal{L}_{aux}$$

$$\mathcal{L}_{aux} = \sum_s w_s\,\tfrac{1}{|\mathcal{A}_s|}\sum_{a\in\mathcal{A}_s}\Big[(\sigma(\hat q_{s,a}) - F1_{s,a})^2 + \mathrm{SmoothL1}\big(\mathrm{softplus}(\hat c_{s,a}),\ \log(1+T_{s,a})\big)\Big]$$

- C는 보정 대상 상태, R은 유지 상태다. π₀는 학습 전 head다.
- 보정 상태에서는 목표를 기존 분포와 a*의 반반으로 둬서 급격한 변화를 막는다.
- 𝓛_aux는 MLP 중간 표현(128차원)에서 행동별 F1과 log 토큰을 예측하는 보조 손실이다. 추론 때는 쓰지 않는다.
- 설정: AdamW 1e-4, 100 epoch.

#### (2) GRPO (head만 학습)

문항마다 G=4개 궤적을 샘플링하고, 궤적 보상 R로 그룹 상대 어드밴티지를 만든다. 그 어드밴티지를 궤적의 모든 결정에 부여한다.

$$A_i = \frac{R_i - \mathrm{mean}(R_{1..G})}{\mathrm{std}(R_{1..G}) + 10^{-6}}$$

$$\mathcal{L} = \sum_{i,t} w_{i,t}\Big[-\min\big(\rho_{i,t}A_i,\ \mathrm{clip}(\rho_{i,t},1\pm0.2)\,A_i\big) + \beta\,\mathrm{KL}\big(\pi_\theta(\cdot|s_{i,t})\,\|\,\pi_{ref}(\cdot|s_{i,t})\big) - \eta\,H(\pi_\theta(\cdot|s_{i,t}))\Big]$$

- ρ = π_θ(a|s)/π_old(a|s)는 행동 확률의 비율이다. LLM 토큰은 포함하지 않는다.
- w는 궤적 길이의 역수 가중치다. 각 궤적이 같은 비중을 갖는다.

| | 9B (GRPO48, 토큰 벌점 비교) | 4B (GRPO96, 공유 코퍼스 GRPO) |
|---|---|---|
| 보상 R | EM + 0.25·F1 − λ·T/(T+256) + 0.02·[형식 준수] | F1 − 0.03·cost |
| 비용 정의 | T = 인자+답의 생성 토큰 수. λ = 0.10, 0.50, 2.0, 10.0 | cost = [도구 호출] + 생성토큰/100 + prefix 토큰/10,000 |
| KL β / 엔트로피 η | 0.02 / 0 | 0.02 / 0.01 |
| 최적화 | AdamW 1e-5, 2 epoch, minibatch 16 | AdamW 1e-4, 4 epoch, minibatch 32 |
| 궤적 | 원래 정책의 192궤적 한 묶음을 모든 λ에 재사용(1회 업데이트) | 반복마다 새로 on-policy 수집 |

## 2. 결과: 정확도는 오르지 않음

### 9B, dev64

| 지표 | 원래 정책 | λ=.10 | λ=.50 | λ=2.0 | λ=10.0 |
|---|---:|---:|---:|---:|---:|
| EM 정답/64 | 25 | 24 | 24 | 24 | 24 |
| F1 (%) | 49.46 | 47.90 | 47.90 | 47.90 | 47.90 |
| 전체 평균 생성토큰 | 158.02 | 161.70 | 161.19 | 161.06 | 155.06 |
| 정답 평균 생성토큰 | 144.92 | 147.71 | 146.33 | 146.17 | 145.21 |

계수를 100배로 키워도 정확도는 그대로였다. 토큰은 λ=10에서만 −1.9% 줄었다.

### 4B

| 실험 | 결과 |
|---|---|
| GRPO96, 새 80문항 | F1: 기준 .321, PPO .343, GRPO .334. 차이 신뢰구간이 0 포함 |
| 공유 코퍼스, dev64 | H3 멈춤 head F1 .273 / EM 14, 기본 GRPO F1 .225 / EM 10 |

## 3. 결과: 토큰 절감은 주로 실패 사례에서

| 비교 | 공통 정답 | 정답 상실 | 둘 다 오답 | 전체 |
|---|---:|---:|---:|---:|
| 9B λ=10 vs 원래 정책 (생성 토큰 절약) | 8 (24문항) | 74 (1문항) | 107 (39문항) | 189 |
| 4B H3 vs 기본 GRPO (H3의 순절약) | 188 (8문항) | 한쪽만 정답 +12 (8문항) | 1,817 (48문항) | 2,017 |

- **9B:** 공통 정답 24문항의 평균은 145.54 → 145.21(−0.2%)이다.
- **4B:** 순절약의 90%가 둘 다 오답인 문항에서 나왔다. H3만 정답인 6문항에서는 148을 더 썼고, GRPO만 정답인 2문항에서는 160을 절약했다.
- **9B 보상 스모크(train 2문항):** 정답 문항은 83→83 토큰 그대로였다. 오답 문항만 같은 오답을 내며 68→23으로 줄었다.

**원인**
- 길이 벌점이 정답 여부와 무관하다. 그래서 가장 쉽게 줄일 수 있는 오답 경로가 짧아진다.
  - λ=10에서는 48개 학습 그룹 중 10개에서 짧은 오답이 정답 경로보다 높은 보상을 받았다.
- 그룹 std 정규화가 길이 차이만 있는 그룹을 ±1 어드밴티지로 증폭하고, λ의 크기 효과는 상쇄한다.
- 벌점이 생성 토큰에 걸려 있다. 실제 비용의 약 99%는 입력 토큰이다(문항당 입력 약 16.8k, 생성 약 158).

## 4. 왜 head는 정책 레버가 못 되나

**도구 선택은 이미 거의 최선이다 (9B, train48 317상태)**

| 분석 | 결과 |
|---|---|
| 한 단계만 다른 행동으로 바꾼 분기 | head 선택 F1 .506 > 대안 평균 .406. 대안이 나은 상태 12.6% |
| oracle 멈춤 + 재구성 답변 | EM 23 → 25 (+2) |
| 학습한 멈춤 probe(AUC .72) | 기준선들을 지배하는 점 없음 |

**병목은 LLM의 읽기와 컨텍스트다 (9B, dev64)**

| 조건 | F1 | EM | 입력 토큰 |
|---|---:|---:|---:|
| gold 근거 2문단만 제공 | 73.8 | 41/64 | 332 |
| 20문단 pool 제공 | 60.4 | 32/64 | 2,517 |
| 에이전트(head) | 49.5 | 25/64 | 16,762 |

head가 바꿀 수 있는 것은 "다음 도구"뿐이다. 정확도를 가르는 것은 LLM의 근거 읽기이고, 토큰을 가르는 것은 매 턴 다시 prefill되는 컨텍스트다. head는 둘 다 건드리지 못한다.

**효과가 보인 곳: 무엇을 남길지 고르기(④, 9B, train48)**

| 조건 | EM /48 | F1 | 답변 컨텍스트 |
|---|---:|---:|---:|
| 에이전트 원래 답 | 21 | .516 | 누적 이력 |
| 본 chunk 전부로 재답변 | 23 | .600 | 약 3.6k |
| 9B 16층 probe 상위 5 chunk로 재답변 | 28 | .700 | 약 900 |
| gold만 (oracle) | 30 | .738 | — |

오프라인 소표본이고, 상위 5개는 사후에 고른 값이다. 새 문항에서는 확인하지 않았다.

## 5. 결론과 다음 방향

1. **head의 역할:** 도구 선택기로 쓴다. 정확도·토큰 정책으로 쓰는 것은 중단한다.
2. **토큰 판정 기준:** 공통 정답 문항의 토큰으로 판정한다. 오답 경로 단축은 절감으로 세지 않는다.
3. **정책 레버 후보:**
   - 컨텍스트에 무엇을 남길지 고르기(④)
   - LLM 자체 학습(LoRA). 계획 `docs/JEV_LORA_GRPO_NATIVE_VS_HEAD_PLAN_20261025.md`, 미실행
4. **보상 설계(LoRA 학습 시):**
   - 비용은 입력 토큰에 둔다.
   - 길이 보상은 정답일 때만 준다.
   - 어드밴티지는 std로 나누지 않는다.

## 출처

| 내용 | 파일 |
|---|---|
| 9B GRPO, 토큰 벌점 비교 | `docs/JEV_FINAL_GRPO_RESULT_20261025.md`, `docs/JEV_GRPO_TOKEN100_RESULT_20261025.md`, `docs/JEV_FINAL_GRPO_REWARD_SMOKE_RESULT_20261025.md` |
| 4B GRPO96 | `docs/JEV_DECISION_GRPO96_REPORT_20261023.md`, `lab/registry/decision_grpo96_20261023/manifest.json` |
| 4B 공유 코퍼스, H3 비교 | `lab/registry/decision_shared_grpo192_20261023/manifest.json`, `lab/reports/decision_h3_role_audit_20261024/correctness_cost.json` |
| 분기, 천장, ④ | `docs/JEV_POC_DECISION_AXIS_REPORT_20261025.md` |
| head 구조와 학습 코드 | `src/lsdrag/decision_state_learning.py`, `src/lsdrag/decision_policy.py`, `scripts/fit_qwen35_bootstrap.py`, `scripts/fit_qwen35_two_promising.py`, `scripts/run_qwen35_grpo_token100.py`, `scripts/run_decision_grpo96.py`, `scripts/run_decision_shared_grpo.py` |
