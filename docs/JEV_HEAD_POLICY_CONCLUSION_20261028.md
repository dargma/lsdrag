# 결론 보고: selection head는 도구 선택은 되지만 정확도·토큰 절감 정책은 안 된다 (2026-10-28)

## 요약

- head는 **도구 선택**은 할 수 있다. 모방 학습만으로 native 에이전트의 행동을 따라 한다.
- head를 **정책**으로 GRPO 학습해도 정확도는 오르지 않았다(F1 22.50 → 22.50). 생성 토큰은 −3%에 그쳤다.
- 모방과 GRPO 모두 LLM이 직접 도구를 고르는 baseline(F1 24.70)보다 낮다.

대상은 Qwen3-4B-Instruct-2507이다. 평가는 반복 사용한 개발 문항(dev64)에서 했으므로 독립 일반화 근거는 아니다.

## 1. 실험 구성

### 1.1 에이전트

- **도구:** keyword_search, semantic_search, read_chunk, FINAL(4개 행동)
- **코퍼스:** MuSiQue train+dev 전체 문단을 중복 제거한 공유 코퍼스 `musique_shared_20261023`
- **종료 규칙:** 도구 호출 8회 이후에는 FINAL을 강제한다.

**매 결정 단계의 흐름**
1. LLM(학습하지 않음)이 현재 컨텍스트를 prefill한다.
2. 행동을 고른다.
   - **baseline:** LLM이 도구 이름을 직접 생성한다(native A-RAG).
   - **head 조건:** head가 LLM hidden으로 고른다.
3. LLM이 그 행동의 인자(또는 답 JSON)를 문법 제약 하에 생성한다.

### 1.2 head 구조 (`PolicyHead`)

| 항목 | 내용 |
|---|---|
| 입력 | LLM 마지막 층 hidden의 마지막 토큰 (2,560차원). 모방 학습 데이터의 평균·표준편차로 정규화 |
| 본체 | trunk Linear(2560→128) → Tanh → action Linear(128→4), skip Linear(2560→4). logit = action 출력 + skip 출력 |
| 파라미터 수 | 약 34만 |
| 출력 | 4개 행동 logit. 불가능한 행동(관측 전 read_chunk 등)은 −10⁹으로 마스크 |

### 1.3 데이터셋

모든 문항은 MuSiQue 2hop이다(`decision_state_policy_iteration_20261023` split). 학습과 평가 문항은 겹치지 않는다.

| 단계 | 데이터 |
|---|---|
| 모방 학습 | train192: native 4B 에이전트가 각 상태에서 고른 행동 |
| GRPO | train192 × 4궤적 × 2회 반복 = 1,536궤적 (반복마다 현재 head로 새로 수집) |
| 평가 | dev64 (세 조건 모두 같은 문항, greedy) |

### 1.4 학습 objective

#### 모방 (BC)

$$\mathcal{L}_{BC} = \sum_{s} w_s\Big[\,0.95\,\big(-\log \pi_\theta(a^{t}_s \mid s)\big) \;+\; 0.05\,\Big(-\frac{1}{|\mathcal{A}_s|}\sum_{a\in\mathcal{A}_s}\log\pi_\theta(a \mid s)\Big)\Big]$$

| 기호 | 의미 |
|---|---|
| s | 결정 상태. native 에이전트 궤적에서 행동을 고르기 직전의 컨텍스트 |
| aᵗₛ | 상태 s에서 native 에이전트(teacher)가 실제로 고른 행동 |
| π_θ(a\|s) | head가 준 행동 확률. 가능한 행동에만 softmax |
| 𝒜ₛ | 상태 s에서 가능한 행동 집합 |
| 첫째 항 | teacher 행동의 음의 로그우도. 같은 행동을 고르도록 학습 |
| 둘째 항 | 가능한 행동 전체에 대한 균등 분포 쪽 항(label smoothing 0.05). 확률이 한 행동에 과도하게 몰리지 않게 함 |
| w_s | 문항마다 상태 가중치 합이 같도록 한 가중치(1/문항 내 상태 수, 전체 합 1). 긴 궤적의 문항이 학습을 지배하지 않게 함 |

설정: trunk·action·skip 학습, AdamW 3e-4, 전체 배치 100 step.

#### GRPO

문항마다 현재 head로 G=4개 궤적을 샘플링한다.

**궤적 보상**

$$R_i = \mathrm{F1}_i - 0.03 \sum_{t} c_{i,t},\qquad c_{i,t} = \mathbf{1}[a_{i,t}\neq\mathrm{FINAL}] + \frac{g_{i,t}}{100} + \frac{p_{i,t}}{10{,}000}$$

| 기호 | 의미 |
|---|---|
| F1ᵢ | 궤적 i의 최종 답 F1 (공식 MuSiQue 채점) |
| c_{i,t} | 결정 t의 비용 |
| 1[a≠FINAL] | 도구 호출이면 1. 호출 횟수에 대한 비용 |
| g_{i,t} | 그 결정에서 생성한 토큰 수(인자 또는 답) |
| p_{i,t} | 그 결정에서 prefill한 입력 토큰 수 |

평가 기준 문항당 비용 합은 약 3.8~3.9였다. 도구 호출 약 2.4 + 생성 약 1.0 + 입력 약 0.5다.

**어드밴티지와 손실**

$$A_i = \frac{R_i - \mathrm{mean}(R_1,\dots,R_4)}{\mathrm{std}(R_1,\dots,R_4) + 10^{-6}}$$

$$\mathcal{L} = \frac{1}{|B|}\sum_{(i,t)\in B} w_{i}\Big[-\min\big(\rho_{i,t}A_i,\ \mathrm{clip}(\rho_{i,t},\,0.8,\,1.2)\,A_i\big) \;+\; 0.02\,\mathrm{KL}\big(\pi_\theta(\cdot|s_{i,t})\,\|\,\pi_{BC}(\cdot|s_{i,t})\big) \;-\; 0.01\,H\big(\pi_\theta(\cdot|s_{i,t})\big)\Big]$$

| 기호 | 의미 |
|---|---|
| Aᵢ | 같은 문항 4개 궤적 안에서 표준화한 상대 보상. 궤적의 모든 결정에 같은 값을 줌. 4개 보상이 모두 같으면 0 |
| B | minibatch(32개 결정). 강제 FINAL 결정은 제외 |
| ρ_{i,t} | π_θ(a\|s) / π_old(a\|s). 수집 당시 head 대비 지금 head의 행동 확률 비율(LLM 토큰 확률은 포함하지 않음) |
| clip(ρ, 0.8, 1.2) | PPO clipping. 한 번에 정책이 크게 바뀌지 않게 함 |
| KL(π_θ ‖ π_BC) | 모방 head 분포에서 멀어지는 정도. 4개 행동에 대해 정확히 계산 |
| H(π_θ) | 행동 분포의 엔트로피. 빼는 항이라 탐색을 유지하는 쪽으로 작용 |
| wᵢ | 궤적 길이의 역수에 비례하는 가중치. 결정이 많은 궤적도 궤적 하나만큼만 반영 |

설정: action·skip만 학습(trunk 고정), AdamW 1e-4, 반복마다 4 epoch, 총 660 update.

## 2. 결과 (dev64)

| 조건 | F1 (%) | 생성 토큰 | 입력 토큰 | 도구 호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| baseline (native, LLM이 도구 선택) | 24.70 | 97.4 | 4,691 | 2.38 | 0 |
| 모방 (BC head) | 22.50 | 101.6 | 4,749 | 2.39 | 0 |
| GRPO 적용 | 22.50 | 98.4 | 4,625 | 2.33 | 0 |

**읽는 법**
- **모방:** baseline의 행동을 따라 하지만 F1은 2.2pp 낮다.
- **GRPO:** 모방 대비 F1 변화가 없다. 생성 토큰 −3.1%, 입력 토큰 −2.6%로 비용만 조금 줄었다.
- **토큰 절감의 정체:** 다른 head와 비교한 기록에서도 토큰 차이는 대부분 오답 문항에서 났다. H3 head는 GRPO보다 2,017토큰을 덜 썼는데, 그중 1,817이 둘 다 오답인 문항이었다.

**GRPO가 정확도 정책이 못 된 원인**
- 보상의 비용 항이 정답 여부와 무관하다. 그래서 가장 쉽게 줄일 수 있는 오답 경로가 짧아진다.
- 그룹 std 정규화가 작은 비용 차이도 ±1 어드밴티지로 키운다.
- head가 바꿀 수 있는 것은 "다음 도구"뿐이다. 정확도를 가르는 근거 읽기(LLM)와 비용 대부분인 입력 컨텍스트는 건드리지 못한다.

## 3. 결론과 다음 방향

1. **head의 역할:** 도구 선택기로 쓴다. 정확도·토큰 정책으로 쓰는 것은 중단한다.
2. **토큰 판정 기준:** 공통 정답 문항의 토큰으로 판정한다.
3. **정책 레버 후보:**
   - 컨텍스트에 무엇을 남길지 고르기: 9B 오프라인 POC에서 EM 21 → 28/48. `docs/JEV_POC_DECISION_AXIS_REPORT_20261025.md`
   - LLM 자체 학습(LoRA): 계획 `docs/JEV_LORA_GRPO_NATIVE_VS_HEAD_PLAN_20261025.md`, 미실행
4. **보상 설계(LoRA 학습 시):**
   - 비용은 입력 토큰에 둔다.
   - 길이 보상은 정답일 때만 준다.
   - 어드밴티지는 std로 나누지 않는다.

## 출처

| 내용 | 파일 |
|---|---|
| 결과 (3조건) | `lab/reports/decision_shared_validation64_20261023/report.md` |
| 설정·데이터 | `lab/registry/decision_shared_grpo192_20261023/manifest.json`, `docs/JEV_DECISION_SHARED_FINAL_REPORT_20261024.md` |
| H3 대 GRPO 토큰 분해 | `lab/reports/decision_h3_role_audit_20261024/correctness_cost.json` |
| 코드 | `src/lsdrag/decision_policy.py`(head, rollout, 비용), `scripts/fit_decision_shared_initialization.py`(모방), `scripts/run_decision_shared_grpo.py`(GRPO) |
