# native A-RAG vs selection head — 둘 다 LoRA + GRPO로 학습해 비교 (계획, 2026-10-25)

## 질문

**목적(사용자):** frozen LLM 위의 head만으로는 정확도 정책을 못 만든다. 정확도는 결국 LLM이 해 줘야 한다. 그렇다면 LLM(LoRA)과 head를 함께 학습할 때 시너지가 나는지 확인한다. 기여가 섞이는 것은 감수한다.

같은 9B, 같은 데이터, 같은 RL 예산으로 학습했을 때, **LLM + head 결합 에이전트**가 **LLM 단독 native A-RAG**보다 정확도나 입력 토큰에서 나은가?

**전체 예산은 6시간이다.** 한 번에 잘 되지 않을 것을 전제로, 고치고 다시 돌릴 시간을 일정에 넣었다.

GPT의 이전 GRPO와 다른 점:
- 이전에는 reader를 얼리고 head만 학습했다.
- 이번에는 두 조건 모두 LLM에 LoRA를 붙여 함께 학습한다.

## 두 조건

| | A. native + LoRA | B. head + LoRA |
|---|---|---|
| 시작점 | Qwen3.5-9B 그대로(LoRA B=0) | 같은 9B + 학습된 head `value_aux.pt`(5191c60e) |
| 행동 선택 | LLM이 도구 이름을 생성 (`fast_native` trie, 샘플링) | head가 pack hidden에서 행동 분포를 만들어 샘플링 |
| 인자·답 생성 | LoRA LLM이 JSON 생성(문법 제약, 샘플링) | 같음 |
| 학습 대상 | LoRA | LoRA + head 동시 |
| 정책 log-prob | 생성 토큰 전체(도구 이름 + 인자 + FINAL 답) | head 행동 logp + 생성 토큰(인자 + 답) |

pack은 마지막 층 hidden에서 만든다(slot 0 last, 1 question-end, 4 observation-mean). 그래서 B에서는 LoRA가 바뀌면 head 입력도 바뀌고, gradient가 head를 지나 LoRA까지 흐른다.

**공통으로 고정하는 것:**
- 도구 3개 + FINAL, 프롬프트 `text_rag_integrated_v5`, common chat template
- 공유 코퍼스 `musique_shared_20261023`
- 최대 8번 도구 호출 후 FINAL 강제, context 16k
- LoRA 설정, 보상, 학습 문항, rollout 수, optimizer step 수

**바닐라에 대해:** 여기서 native는 FINAL을 JSON 도구처럼 쓰는 우리 변형이다. 공식 A-RAG처럼 FINAL이 없고 텍스트 응답이 종료인 바닐라는 이번 범위에서 제외한다. 그래야 두 조건의 차이가 "누가 행동을 고르나" 하나로 남는다. 필요하면 평가 때만 바닐라를 추가한다.

## 학습 설정 (두 조건 동일)

| 항목 | 값 | 이유 |
|---|---|---|
| LoRA | rank 16, α 32, dropout 0. 대상: full-attention 층 q/k/v/o + 모든 MLP(gate/up/down). linear-attention(DeltaNet) 층은 MLP만 | 커널 의존을 줄이고, 표준적인 설정 |
| 학습률 | LoRA 1e-5, head 1e-5 (AdamW, grad clip 1.0) | GPT head GRPO와 같은 lr |
| 문항 | train192(`decision_state_policy_iteration_20261023` train). 반복마다 비복원으로 뽑고, 두 조건이 같은 순서를 쓴다 | dev64·reserved80 사용 안 함 |
| 그룹 | 문항당 4개 궤적, 온도 1.0 | |
| 반복 | 조건당 75분 안에서 가능한 만큼. 반복마다 1 epoch(minibatch 16 결정) | S0 속도로 반복 크기 결정 |
| 손실 | PPO clip ±0.2, KL β=0 | Dr.GRPO/DAPO 방식. ref forward를 없애 비용 절감 |
| 어드밴티지 | r − 그룹평균 (std로 나누지 않음) | GPT 실험에서 z-score가 토큰 벌점을 무력화했던 문제 회피 |

**보상:** R = EM + 0.25·F1 − 0.2·(입력토큰/20,000) − 0.5·[형식오류]
- 비용 항은 생성 토큰이 아니라 **입력 토큰**에 둔다. 실제 비용의 99%가 입력이다(GPT는 생성 토큰에 벌점을 줬다).
- 문항당 입력은 평균 약 17k이므로 벌점은 약 −0.17이다. 정답 1개(+1.0~1.25)보다 확실히 작다.

## 6시간 일정 (시간 상자)

| 시각 | 단계 | 끝나는 조건 |
|---|---|---|
| 0:00–0:45 | S0 환경·속도 | 측정값으로 S3 규모 확정 |
| 0:45–1:45 | S1 구현 + S2 파일럿 (A, B 각각) | 파일럿 확인 항목 통과 |
| 1:45–2:00 | 수정 여유 1 | 파일럿 실패 항목만 수정 후 재파일럿 |
| 2:00–4:30 | S3 본 학습 A → B (각 75분, 시간 기준) | 시간이 되면 그 반복에서 끝냄 |
| 4:30–5:40 | S4 평가 | 문항 수는 S0 속도에 맞춤 |
| 5:40–6:00 | 보고서·push | |

**원칙:**
- 학습은 반복 횟수가 아니라 **시간**(조건당 75분)으로 같게 맞춘다. 두 조건의 rollout 수가 달라질 수 있으므로 함께 보고한다.
- 문제가 생기면 그 단계 안에서 고친다. 일정을 넘기면 규모를 줄인다(문항 수 → 반복 수 순서). 설정은 바꾸지 않는다.
- 각 단계의 결과와 실패는 바로 메모리에 기록한다.

### S0 환경·속도 (45분)
- `flash-linear-attention`과 `causal-conv1d`를 설치한다. 지금은 torch fallback이라 느리고, DeltaNet 역전파는 특히 느리다.
  - 설치에 실패하거나 출력이 달라지면 fallback으로 진행하고 규모를 줄인다.
- 설치 전후로 같은 입력의 logits 차이를 확인한다(bf16 허용 오차 이내).
- LoRA(B=0)를 붙인 모델이 원래 모델과 같은 출력을 내는지 확인한다.
- 측정: 에피소드 1개 rollout 시간, 학습 forward+backward 시간과 최대 메모리(gradient checkpointing).
- 이 측정으로 조건당 75분에 들어가는 반복 크기를 정한다. 예: 에피소드 30초면 반복 1회 = 16문항 × 4궤적 ≈ 32분+학습 → 반복 2회.

### S1 구현 (`src/lsdrag/lora_grpo.py`, `scripts/run_lora_grpo.py`)
1. **샘플링 rollout**
   - `rollout()`에 `sample_tokens=True`를 추가한다.
   - 도구 이름(trie), 인자, 답 토큰을 온도 1로 샘플링한다. 문법 마스크로 막힌 토큰이 나오면 허용 집합 안에서 다시 샘플링한다.
   - 각 결정마다 생성 토큰 id, 생성 시점 logp, 결정 위치를 저장한다.
2. **학습용 log-prob 재계산**
   - 결정마다 그 시점의 prefix + 생성 토큰으로 teacher-forcing forward(LoRA, grad on)를 해서 토큰 logp를 계산한다. B는 같은 forward에서 pack → head logp도 계산한다.
   - 문법 마스크 없는 log-softmax를 쓴다. 이 근사와 오차는 기록한다.
   - **replay 확인:** 업데이트 전 재계산 logp와 rollout 때 logp의 최대 차이를 남긴다. 기준: 생성 토큰 평균 |Δ| < 0.05.
3. **저장:** 반복마다 adapter와 head checkpoint, rollout 원본(에피소드 JSON, 보상, 어드밴티지)을 저장한다.

### S2 파일럿 (S1과 합쳐 60분)
- 조건마다 4문항 × 4궤적 × 1업데이트.
- **확인:**
  - loss가 유한한가
  - replay 오차 기준 통과
  - clip 비율 < 30%
  - 업데이트 후 LoRA·head 파라미터가 실제로 바뀌었나
  - 형식 오류율
  - 보상 분산이 있는 그룹 수
- **실패 시 대응(수정 여유 1에서):**
  - 보상 분산 없음 → 그룹 6개로 늘린다
  - 형식 오류 많음 → 샘플링을 인자에서만 온도 0.7로 낮춘다
  - replay 오차 큼 → 문법 마스크를 반영한 logp로 바꾼다

### S3 본 학습 (조건당 75분)
- A → B 순서. 문항 순서는 같다.
- 반복마다 train 보상 평균, EM, 입력 토큰, 형식 오류, 행동 분포(FINAL 비율, 호출 수)를 기록한다.
- 중간 선택이나 조기 종료는 하지 않는다. 마지막 checkpoint로 평가한다.

### S4 평가 (70분)
- 새 문항(`lab/registry/poc_idea4_fresh48_20261026`, 처음 쓰는 dev 2hop)에서 greedy로 평가한다.
- 문항 수는 S0 속도에 맞춘다(목표 48, 최소 24).

| 조건 | 설명 |
|---|---|
| A0 | native, LoRA 없음 |
| A1 | native + LoRA-GRPO |
| B0 | head(value_aux), LoRA 없음 |
| B1 | head + LoRA-GRPO |
| B1-native (시간 되면) | B1의 LoRA를 쓰되 행동은 LLM이 고름 |

B1-native는 학습 없이 B1의 이득이 LoRA에서 왔는지 head에서 왔는지 대략 나눠 본다.

- **지표:** EM(개수, %), F1, 문항당 입력 토큰, 생성 토큰, 호출 수, 형식 오류. 공통 정답 문항의 입력 비교, 문항별 승패.
- **핵심 비교:** A1 vs B1(시너지).
- **참고 비교:** A0→A1, B0→B1 (학습 효과).

## 판정 (대략)
- **B1이 낫다:** EM이 같거나 높고, 입력이 10% 이상 적거나, F1이 +3pp 이상.
- **A1이 낫다:** 반대 경우.
- **차이 없음:** 그 밖.
- n=48이라 문항별 승패를 함께 보고한다. 일반화 주장은 하지 않는다.

## 위험과 대응

| 위험 | 대응 |
|---|---|
| fla 미설치로 학습이 너무 느림 | S0에서 설치·측정. 안 되면 문항·반복 축소 |
| GPU 공유(다른 작업이 100% 사용 중) | 다른 작업은 건드리지 않음. 시간 추정은 S0 실측 기준 |
| A가 zero-shot 시작, B는 사전학습 head로 시작 | 사용자 지정 설계. 출발점 차이는 A0/B0로 같이 보고 |
| 문법 마스크로 인한 off-policy 오차 | replay 오차를 기록. 크면 마스크 반영 logp로 전환 |
| B에서 LoRA가 hidden을 바꿔 head 입력 분포가 이동 | head도 함께 학습. 반복별 행동 분포 기록 |
| 32문항 × 4로 보상 분산 부족 | S2에서 확인. 분산 없는 그룹 비율 기록 |

## 산출물
- 계획: 이 문서
- registry: `lab/registry/lora_grpo_native_vs_head_20261025/`(plan.json, 문항 순서, 해시)
- 결과: `results/lora_grpo_{native,head}_20261025/`, 평가 `results/lora_grpo_eval_fresh48_20261025/`
- 보고서: 전후 표
