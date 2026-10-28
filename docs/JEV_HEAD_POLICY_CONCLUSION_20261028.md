# 결론 보고: selection head는 도구 선택은 되지만 정확도·토큰 절감 정책은 안 된다 (2026-10-28)

## 요약

- head는 native 에이전트의 **도구 선택을 따라 할 수 있다**(학습 상태 기준 행동 일치 99.85%).
- head를 **GRPO로 정책 학습**해도 정확도는 그대로였다(F1 22.50 → 22.50). 생성 토큰은 −3%에 그쳤다.
- 두 head 모두 LLM이 직접 도구를 고르는 baseline(F1 24.70)보다 낮다.

대상: Qwen3-4B-Instruct-2507. 평가: 반복 사용한 개발 문항 dev64(독립 일반화 근거 아님).

## 1. 실험 구성

**에이전트**
- 도구: keyword_search, semantic_search, read_chunk, FINAL
- 코퍼스: MuSiQue train+dev 전체 문단(중복 제거)
- 도구 호출 8회 이후에는 FINAL을 강제한다.
- 매 단계 LLM(학습 안 함)이 컨텍스트를 prefill한다. 행동은 baseline에서는 LLM이 도구 이름을 생성하고, head 조건에서는 head가 고른다. 인자와 답은 항상 LLM이 생성한다.

**head (`PolicyHead`, 약 34만 파라미터)**
- 입력: LLM 마지막 층 마지막 토큰 hidden(2,560차원, 정규화)
- 본체: Linear(2560→128) → Tanh → Linear(128→4), 여기에 skip Linear(2560→4)를 더한다.
- 불가능한 행동은 마스크한다.

**데이터** (MuSiQue 2hop, 학습·평가 문항 비중복)

| 단계 | 데이터 |
|---|---|
| 모방 | train192: native 에이전트의 667개 결정 상태와 행동 |
| GRPO | train192 × 4궤적 × 2회 반복 = 1,536궤적(반복마다 새로 수집) |
| 평가 | dev64, greedy |

## 2. 학습 objective

### 모방 (BC)

$$\mathcal{L}_{BC} = \sum_{s} w_s\Big[0.95 \cdot \big(-\log \pi_\theta(a^{t}_s \mid s)\big) + 0.05 \cdot \Big(-\frac{1}{|\mathcal{A}_s|}\sum_{a\in\mathcal{A}_s}\log\pi_\theta(a \mid s)\Big)\Big]$$

| 기호 | 의미 |
|---|---|
| s, aᵗₛ | native 에이전트의 결정 상태와 그때 고른 행동 |
| π_θ(a\|s) | head의 행동 확률(가능한 행동에만 softmax) |
| 𝒜ₛ | 상태 s에서 가능한 행동 집합 |
| 첫째 항 | teacher 행동의 음의 로그우도 |
| 둘째 항 | label smoothing(0.05). 확률이 한 행동에 과도하게 몰리지 않게 함 |
| w_s | 1/(문항 내 상태 수), 전체 합 1. 긴 궤적의 문항이 학습을 지배하지 않게 함 |

AdamW 3e-4, 전체 배치 100 step.

### GRPO

문항마다 현재 head로 4개 궤적을 샘플링한다.

$$R_i = \mathrm{F1}_i - 0.03 \sum_{t} c_{i,t},\qquad c_{i,t} = \mathbf{1}[a_{i,t}\neq\mathrm{FINAL}] + \frac{g_{i,t}}{100} + \frac{p_{i,t}}{10{,}000}$$

$$A_i = \frac{R_i - \mathrm{mean}(R_{1..4})}{\mathrm{std}(R_{1..4}) + 10^{-6}}$$

$$\mathcal{L} = \frac{1}{|B|}\sum_{(i,t)\in B} w_{i}\Big[-\min\big(\rho_{i,t} A_i,\ \mathrm{clip}(\rho_{i,t},0.8,1.2) \cdot A_i\big) + 0.02 \cdot \mathrm{KL}\big(\pi_\theta \parallel \pi_{BC}\big) - 0.01 \cdot H(\pi_\theta)\Big]$$

| 기호 | 의미 |
|---|---|
| F1ᵢ | 궤적 i의 최종 답 F1(공식 MuSiQue 채점) |
| c_{i,t} | 결정 t의 비용: 도구 호출이면 1 + 생성 토큰 g/100 + 입력 토큰 p/10,000. 문항당 합은 약 3.9(호출 2.4, 생성 1.0, 입력 0.5) |
| Aᵢ | 같은 문항 4궤적 안에서 표준화한 상대 보상. 궤적의 모든 결정에 같은 값을 줌 |
| B | minibatch 32개 결정(강제 FINAL 제외) |
| ρ_{i,t} | π_θ(a\|s)/π_old(a\|s). 수집 당시 대비 행동 확률 비율(LLM 토큰 확률 미포함) |
| clip | PPO clipping. 한 번의 정책 변화를 제한 |
| KL(π_θ‖π_BC) | 모방 head에서 멀어지는 정도에 대한 벌점(4개 행동에 대해 정확히 계산) |
| H(π_θ) | 행동 엔트로피. 빼는 항이라 탐색을 유지 |
| wᵢ | 궤적 길이의 역수 비례. 궤적마다 같은 비중 |

action·skip만 학습(trunk 고정), AdamW 1e-4, 반복마다 4 epoch, 총 660 update.

## 3. 결과 (dev64)

| 조건 | F1 (%) | 생성 토큰 | 입력 토큰 | 도구 호출 |
|---|---:|---:|---:|---:|
| baseline (native) | 24.70 | 97.4 | 4,691 | 2.38 |
| 모방 (BC head) | 22.50 | 101.6 | 4,749 | 2.39 |
| GRPO 적용 | 22.50 | 98.4 | 4,625 | 2.33 |

세 조건 모두 오류는 0이다.

- **모방은 된다:** 학습 상태 667개 중 666개(99.85%)에서 native와 같은 행동을 골랐다. 학습에 쓴 상태 기준이라 일반화 정확도는 아니다. dev64 도구 호출 수도 baseline과 같은 수준이다(2.39 vs 2.38).
- **GRPO는 정확도를 바꾸지 못했다:** F1이 같고, 토큰만 약 3% 줄었다.
- **줄어든 토큰은 주로 오답 문항:** H3 head와 GRPO를 비교한 기록에서 토큰 차이 2,017 중 1,817이 둘 다 오답인 문항에서 났다.

**원인**
- 비용 항이 정답 여부와 무관해서, 가장 쉽게 줄일 수 있는 오답 경로가 짧아진다. 그룹 std 정규화는 작은 비용 차이도 ±1 어드밴티지로 키운다.
- head가 바꾸는 것은 "다음 도구"뿐이다. 정확도를 가르는 근거 읽기(LLM)와 비용 대부분인 입력 컨텍스트는 건드리지 못한다.

## 4. 다음 방향

1. head는 도구 선택기로만 쓰고, 토큰 절감은 공통 정답 문항 기준으로 판정한다.
2. 정책 레버는 두 가지로 옮긴다.
   - 컨텍스트에 무엇을 남길지 고르기: 9B 오프라인 POC에서 EM 21 → 28/48(`jev_v1.1` 브랜치 기록)
   - LLM LoRA 학습: 계획만 있고 미실행(`jev_v1.1` 브랜치)
3. LoRA 학습 보상은 입력 토큰 비용, 정답일 때만 길이 보상, std로 나누지 않는 어드밴티지로 설계한다.

## 출처

| 내용 | 파일 |
|---|---|
| 결과 | `lab/reports/decision_shared_validation64_20261023/report.md` |
| 모방 정확도 | BC 학습 manifest의 `train_accuracy` .998501(`jev_v1.1`의 `memory/CURRENT.md`에 기록), `lab/reports/decision_shared_teacher192_20261023/audit.json` |
| 설정 | `lab/registry/decision_shared_grpo192_20261023/manifest.json` |
| H3 대 GRPO | `lab/reports/decision_h3_role_audit_20261024/correctness_cost.json` |
| 코드 | `src/lsdrag/decision_policy.py`, `scripts/fit_decision_shared_initialization.py`, `scripts/run_decision_shared_grpo.py` |
| 재현 | `docs/REPRODUCE_HEAD_POLICY.md` |
