# 결론 보고: selection head는 도구 선택은 되지만 정확도·토큰 절감 정책은 안 된다 (2026-10-28)

## 요약

- head는 **도구 선택**은 잘한다. 고른 행동이 대안보다 낫고, 형식 오류도 없다.
- head를 **정책**으로 GRPO 학습해도 정확도는 오르지 않았고, 토큰도 줄지 않았다.
- 원인: 정확도는 LLM의 근거 읽기가, 토큰은 매 턴 다시 prefill되는 입력이 결정한다. head는 둘 다 건드리지 못한다.

대상은 Qwen3.5-9B다. 평가는 반복 사용한 개발 문항(dev64)에서 했으므로 독립 일반화 근거는 아니다. 4B에서도 같은 경향이었다(F1 22.50 → 22.50).

## 1. 실험 구성

### 1.1 에이전트

- **도구:** keyword_search, semantic_search, read_chunk, FINAL(4개 행동)
- **코퍼스:** MuSiQue train+dev 전체 문단을 중복 제거한 공유 코퍼스 `musique_shared_20261023`
- **종료 규칙:** 도구 호출 8회 이후에는 FINAL을 강제한다.

**매 결정 단계의 흐름**
1. LLM(Qwen3.5-9B, 학습하지 않음)이 컨텍스트를 prefill한다.
2. head가 그 hidden으로 행동을 고른다.
3. LLM이 인자(또는 답 JSON)를 문법 제약 하에 생성한다.

### 1.2 head 구조 (`StateHead`)

| 항목 | 내용 |
|---|---|
| 입력 | LLM 마지막 층 hidden의 3곳: 마지막 토큰, 질문 끝 토큰, 최근 관측 구간 평균 |
| 입력 차원 | 3 × 4096 = 12,288 (학습 데이터 평균·표준편차로 정규화) |
| 본체 | Linear(12288→128) → Tanh → Linear(128→4) |
| 파라미터 수 | 약 157만 |
| 출력 | 4개 행동 logit. 불가능한 행동(관측 전 read_chunk 등)은 −10⁹으로 마스크 |

### 1.3 데이터셋

모든 문항은 MuSiQue 2hop이다(`decision_state_policy_iteration_20261023` split). 학습과 평가 문항은 겹치지 않는다.

| 단계 | 데이터 |
|---|---|
| ① 모방 학습 | train192: native 9B 에이전트가 고른 행동 |
| ② 결과 기반 보정 | train48(train192의 부분집합): 상태마다 네 행동을 실행해 본 결과 |
| ③ GRPO | train48 × 4궤적 = 192궤적, 결정 880개 |
| 평가 | dev64 (학습 전후 같은 문항) |

### 1.4 학습 objective

**① 모방:** native 에이전트의 행동 aᵗ를 따라 한다.

$$\mathcal{L}_{BC} = \sum_s w_s\big[\,0.95\,(-\log \pi_\theta(a^t_s|s)) + 0.05\,(-\tfrac{1}{|\mathcal{A}_s|}\textstyle\sum_{a\in\mathcal{A}_s}\log\pi_\theta(a|s))\big]$$

- 두 번째 항은 가능한 행동 𝒜ₛ에 대한 label smoothing이다.
- wₛ는 문항별로 합이 같도록 맞춘 가중치다.
- 설정: AdamW 3e-4, 100 step.

그 뒤 ② train48 분기 결과로 한 번 보정한 head(value_aux)를 GRPO 시작점으로 쓴다(보정 목적식은 `scripts/fit_qwen35_two_promising.py`).

**③ GRPO:** 문항마다 4개 궤적을 샘플링하고, 궤적 보상으로 그룹 상대 어드밴티지를 만들어 궤적의 모든 결정에 준다.

$$R = \mathrm{EM} + 0.25\,\mathrm{F1} - 0.10\,\frac{T}{T+256} + 0.02\,[\text{형식 준수}],\qquad A_i = \frac{R_i - \mathrm{mean}(R)}{\mathrm{std}(R) + 10^{-6}}$$

$$\mathcal{L} = \sum_{i,t} w_{i,t}\Big[-\min\big(\rho_{i,t}A_i,\ \mathrm{clip}(\rho_{i,t},1\pm0.2)\,A_i\big) + 0.02\,\mathrm{KL}\big(\pi_\theta\,\|\,\pi_{ref}\big)\Big]$$

- T는 인자와 답의 생성 토큰 수다.
- ρ = π_θ(a|s)/π_old(a|s)는 행동 확률의 비율이다(LLM 토큰 미포함).
- w는 궤적 길이의 역수 가중치다.
- 설정: AdamW 1e-5, 2 epoch, minibatch 16(110 update).

## 2. 결과 (dev64)

| 조건 | F1 (%) | EM /64 | 생성 토큰 | 입력 토큰 |
|---|---:|---:|---:|---:|
| head, GRPO 전 | 49.46 | 25 | 158 | 16,762 |
| head, GRPO 후 | 47.90 | 24 | 162 | 17,122 |
| 참고: 20문단 pool을 직접 제공 | 60.4 | 32 | — | 2,517 |
| 참고: gold 근거 2문단만 제공 | 73.8 | 41 | — | 332 |

**읽는 법**
- **GRPO 효과 없음:** 정확도가 조금 떨어지고 토큰은 늘었다. 정답 여부로 나눠도 모든 묶음에서 토큰이 늘었다.
- **벌점을 키워도 같음:** 토큰 벌점을 100배로 키우면 생성 토큰이 −1.9% 줄었지만, 절약 189 중 181이 오답 문항에서 나왔다.
- **병목은 head 밖:** 같은 LLM도 근거만 깔끔하게 주면 EM 41까지 간다. 정확도를 가르는 것은 LLM의 근거 읽기이고, 토큰을 가르는 것은 매 턴 다시 prefill되는 입력이다(입력이 비용의 약 99%). head는 둘 다 건드리지 못한다.
- **도구 선택은 이미 거의 최선:** train48 분기 분석에서 대안 행동이 나은 상태는 12.6%뿐이었다.
- **효과가 보인 곳:** 남길 근거를 고르는 것(④)이다. 9B 16층 probe로 상위 5 chunk만 남겨 다시 답하면 train48에서 EM 21 → 28이었다. 오프라인 소표본이고 새 문항에서는 확인하지 않았다.

**GRPO가 실패한 원인**
- 길이 벌점이 정답 여부와 무관해서, 가장 쉽게 줄일 수 있는 오답 경로가 짧아진다.
- 그룹 std 정규화가 벌점 계수의 크기 효과를 상쇄한다.
- 벌점이 실제 비용인 입력 토큰이 아니라 생성 토큰에 걸려 있다.

## 3. 결론과 다음 방향

1. **head의 역할:** 도구 선택기로 쓴다. 정확도·토큰 정책으로 쓰는 것은 중단한다.
2. **토큰 판정 기준:** 공통 정답 문항의 토큰으로 판정한다.
3. **정책 레버 후보:**
   - 컨텍스트에 무엇을 남길지 고르기(④)
   - LLM 자체 학습(LoRA). 계획 `docs/JEV_LORA_GRPO_NATIVE_VS_HEAD_PLAN_20261025.md`, 미실행
4. **보상 설계(LoRA 학습 시):** 비용은 입력 토큰에 두고, 길이 보상은 정답일 때만 주고, 어드밴티지는 std로 나누지 않는다.

## 출처

| 내용 | 파일 |
|---|---|
| GRPO 결과 | `docs/JEV_FINAL_GRPO_RESULT_20261025.md`, `docs/JEV_GRPO_TOKEN100_RESULT_20261025.md` |
| 분기, 천장, ④ | `docs/JEV_POC_DECISION_AXIS_REPORT_20261025.md` |
| head와 학습 코드 | `src/lsdrag/decision_state_learning.py`, `scripts/fit_qwen35_bootstrap.py`, `scripts/fit_qwen35_two_promising.py`, `scripts/run_qwen35_final_grpo48.py` |
| 4B 참고 | `lab/reports/decision_shared_validation64_20261023/report.md` |
