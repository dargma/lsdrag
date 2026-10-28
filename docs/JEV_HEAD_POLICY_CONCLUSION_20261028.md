# 결론 보고: selection head는 도구 선택은 되지만 정확도·토큰 절감 정책은 안 된다 (2026-10-28)

## 요약

- head는 **도구 선택**은 잘한다. 고른 행동이 대안보다 낫고, 형식 오류도 없다.
- head를 **정책**으로 GRPO 학습해도 정확도는 오르지 않았다. 4B와 9B의 대표 설정 모두 그렇다.
- 토큰도 의미 있게 줄지 않았다. 9B는 오히려 늘었고, 4B는 −3%였다.
- 다른 설정에서 토큰이 줄었던 경우는 대부분 실패 사례(오답 경로가 짧아짐)였다.

모든 평가는 반복 사용한 개발 문항(dev64)에서 했다. 독립 일반화 근거는 아니다.

## 1. 실험 구성

### 1.1 에이전트

- **도구:** keyword_search, semantic_search, read_chunk, FINAL(4개 행동)
- **코퍼스:** MuSiQue train+dev 전체 문단을 중복 제거한 공유 코퍼스 `musique_shared_20261023`
- **종료 규칙:** 도구 호출 8회 이후에는 FINAL을 강제한다.

**매 결정 단계의 흐름**
1. LLM이 현재 컨텍스트를 prefill한다.
2. head가 그 hidden으로 행동을 고른다.
3. LLM이 그 행동의 인자(또는 답 JSON)를 문법 제약 하에 생성한다.

**LLM은 학습하지 않는다.** head만 학습했다.

### 1.2 head 구조

| | 9B head (`StateHead`) | 4B head (`PolicyHead`) |
|---|---|---|
| LLM | Qwen3.5-9B (hidden 4096) | Qwen3-4B-Instruct-2507 (hidden 2560) |
| 입력 위치 | 마지막 층 hidden의 3곳: 마지막 토큰, 질문 끝 토큰, 최근 관측 구간 평균 | 마지막 층 hidden의 마지막 토큰 1곳 |
| 입력 차원 | 3 × 4096 = 12,288 (학습 데이터 평균·표준편차로 정규화) | 2,560 (정규화) |
| 본체 | Linear(12288→128) → Tanh → Linear(128→4) | trunk Linear(2560→128) → Tanh → action Linear(128→4), skip Linear(2560→4) |
| 파라미터 수 | 약 157만 | 약 34만 |
| 출력 | 4개 행동 logit. 불가능한 행동(관측 전 read_chunk 등)은 −10⁹으로 마스크 | 같음 |
| GRPO 학습 대상 | MLP 전체(정규화 통계 고정) | action, skip만 (trunk 고정) |

### 1.3 데이터셋

모든 문항은 MuSiQue 2hop이다. 4B와 9B는 같은 split(`decision_state_policy_iteration_20261023`)을 쓰고, 학습과 평가 문항은 겹치지 않는다.

| 단계 | 9B | 4B |
|---|---|---|
| head 사전학습 | train192: native 9B가 고른 행동(모방) → train48: 상태마다 네 행동을 실행해 본 분기 결과로 보정 | train192: native 4B가 고른 행동(모방) |
| GRPO 학습 | train48 × 4궤적 = 192궤적, 결정 880개, 1회 수집 | train192 × 4궤적 × 2회 반복 = 1,536궤적 |
| 평가 | dev64 (학습 전후 같은 문항) | dev64 (같음) |

### 1.4 학습 objective

#### (1) head 사전학습

**모방(BC):** native 에이전트가 고른 행동 aᵗ를 따라 하도록 학습한다. 9B와 4B 모두 이 단계를 거친다.

$$\mathcal{L}_{BC} = \sum_s w_s\big[\,0.95\,(-\log \pi_\theta(a^t_s|s)) + 0.05\,(-\tfrac{1}{|\mathcal{A}_s|}\textstyle\sum_{a\in\mathcal{A}_s}\log\pi_\theta(a|s))\big]$$

- 두 번째 항은 가능한 행동 𝒜ₛ에 대한 label smoothing이다.
- wₛ는 문항별로 합이 같도록 맞춘 가중치다. 상태가 많은 문항이 학습을 지배하지 않게 한다.
- 설정(9B, 4B 동일): AdamW 3e-4, 100 step.

**결과 기반 보정(9B만, value_aux):** 각 상태에서 네 행동을 모두 실행해 본 결과(F1, 생성 토큰)로 목표 행동 a*를 정한다.
- a*의 조건: 정답성(EM, F1)이 현재 행동 이상이고, F1이 더 높거나 토큰이 더 적다.

$$\mathcal{L} = \tfrac12\sum_{s\in C} w_s\,\mathrm{CE}\big(\tfrac12\pi_0(\cdot|s)+\tfrac12\,\mathbf{1}_{a^*_s},\ \pi_\theta\big) + \tfrac12\sum_{s\in R} w_s\,\mathrm{CE}\big(\pi_0(\cdot|s),\ \pi_\theta\big) + 0.1\,\mathcal{L}_{aux}$$

$$\mathcal{L}_{aux} = \sum_s w_s\,\tfrac{1}{|\mathcal{A}_s|}\sum_{a\in\mathcal{A}_s}\Big[(\sigma(\hat q_{s,a}) - F1_{s,a})^2 + \mathrm{SmoothL1}\big(\mathrm{softplus}(\hat c_{s,a}),\ \log(1+T_{s,a})\big)\Big]$$

- C는 보정 대상 상태, R은 유지 상태다. π₀는 보정 전 head다.
- 보정 상태에서는 목표를 기존 분포와 a*의 반반으로 둬서 급격한 변화를 막는다.
- 𝓛_aux는 MLP 중간 표현(128차원)에서 행동별 F1과 log 토큰을 예측하는 보조 손실이다. 추론 때는 쓰지 않는다.
- 설정: AdamW 1e-4, 100 epoch.

#### (2) GRPO (head만 학습)

문항마다 G=4개 궤적을 샘플링하고, 궤적 보상 R로 그룹 상대 어드밴티지를 만든다. 그 어드밴티지를 궤적의 모든 결정에 부여한다.

$$A_i = \frac{R_i - \mathrm{mean}(R_{1..G})}{\mathrm{std}(R_{1..G}) + 10^{-6}}$$

$$\mathcal{L} = \sum_{i,t} w_{i,t}\Big[-\min\big(\rho_{i,t}A_i,\ \mathrm{clip}(\rho_{i,t},1\pm0.2)\,A_i\big) + \beta\,\mathrm{KL}\big(\pi_\theta(\cdot|s_{i,t})\,\|\,\pi_{ref}(\cdot|s_{i,t})\big) - \eta\,H(\pi_\theta(\cdot|s_{i,t}))\Big]$$

- ρ = π_θ(a|s)/π_old(a|s)는 행동 확률의 비율이다. LLM 토큰은 포함하지 않는다.
- w는 궤적 길이의 역수 가중치다. 각 궤적이 같은 비중을 갖는다.

| | 9B | 4B |
|---|---|---|
| 시작 head | value_aux | BC |
| 보상 R | EM + 0.25·F1 − 0.10·T/(T+256) + 0.02·[형식 준수] | F1 − 0.03·cost |
| 비용 정의 | T = 인자+답의 생성 토큰 수 | cost = [도구 호출] + 생성토큰/100 + prefix 토큰/10,000 |
| KL β / 엔트로피 η | 0.02 / 0 | 0.02 / 0.01 |
| 최적화 | AdamW 1e-5, 2 epoch, minibatch 16 (110 update) | AdamW 1e-4, 4 epoch, minibatch 32 (660 update) |

## 2. 결과: GRPO 전후 (dev64)

| 지표 | 9B 전 | 9B 후 | 4B 전 | 4B 후 |
|---|---:|---:|---:|---:|
| F1 (%) | 49.46 | 47.90 | 22.50 | 22.50 |
| EM 정답/64 | 25 | 24 | — | 10 |
| 평균 생성 토큰 | 158.02 | 161.70 | 101.59 | 98.41 |
| 평균 입력 토큰 | 16,762 | 17,122 | 4,749 | 4,625 |
| 평균 도구 호출 | 5.55 | 5.62 | 2.39 | 2.33 |
| 오류 | 0 | 0 | 0 | 0 |

- **9B:** 정확도가 떨어지고(EM −1) 토큰은 늘었다. 정오별 토큰은 모든 묶음에서 늘었다.

  | 문항 구분 | 문항 수 | 전→후 총 생성 토큰 |
  |---|---:|---|
  | 둘 다 정답 | 24 | 3,443 → 3,545 |
  | 정답 → 오답 | 1 | 180 → 281 |
  | 둘 다 오답 | 39 | 6,490 → 6,523 |

- **4B:** F1은 같고 생성 토큰만 −3.1%다. 학습 전 EM은 기록에 없다.

**참고 (대표 설정 밖):** 토큰이 줄었던 다른 설정은 절약 대부분이 실패 사례였다.
- 9B 생성 토큰 벌점을 100배로 키운 설정: 절약 189 중 둘 다 오답 107, 정답 상실 74, 공통 정답 8
- 4B H3 head vs GRPO: 순절약 2,017 중 둘 다 오답 1,817

**원인**
- 길이 벌점이 정답 여부와 무관해서, 가장 쉽게 줄일 수 있는 오답 경로가 짧아진다.
- 그룹 std 정규화는 길이 차이만 있는 그룹을 ±1 어드밴티지로 증폭하고, 벌점 계수의 크기 효과는 상쇄한다.
- 벌점이 생성 토큰에 걸려 있다. 실제 비용의 약 99%는 입력 토큰이다(9B 문항당 입력 약 16.8k, 생성 약 158).

## 3. 왜 head는 정책 레버가 못 되나

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

## 4. 결론과 다음 방향

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
| 9B GRPO (대표) | `docs/JEV_FINAL_GRPO_RESULT_20261025.md`, `lab/registry/qwen35_final_grpo48_20261025/` |
| 4B GRPO (대표) | `lab/reports/decision_shared_validation64_20261023/report.md`, `lab/registry/decision_shared_grpo192_20261023/manifest.json`, `docs/JEV_DECISION_SHARED_FINAL_REPORT_20261024.md` |
| 참고: 9B 벌점 비교, 4B H3 비교 | `docs/JEV_GRPO_TOKEN100_RESULT_20261025.md`, `lab/reports/decision_h3_role_audit_20261024/correctness_cost.json` |
| 분기, 천장, ④ | `docs/JEV_POC_DECISION_AXIS_REPORT_20261025.md` |
| head 구조와 학습 코드 | `src/lsdrag/decision_state_learning.py`, `src/lsdrag/decision_policy.py`, `scripts/fit_qwen35_bootstrap.py`, `scripts/fit_qwen35_two_promising.py`, `scripts/run_qwen35_final_grpo48.py`, `scripts/run_decision_shared_grpo.py` |
