# 선정 head의 GRPO 추가 학습 결과

## 결론

이미 outcome 학습된 다중 위치 MLP head 자체를 GRPO로 추가 학습했다. 입력 정밀도 손실로 중단된 첫 실행을 보존하고, FP32 상태 저장으로 확률 재계산 오류를 해결했다. 검사 허용오차는 낮추지 않았다. 두 라운드 학습·전체 감사·검증64 전후 비교·원자료와 checkpoint 백업을 완료했다.

검증64에서 추가 GRPO는 선정 head의 생성 토큰을 30.70% 줄였지만 F1 평균은 2.23%p 낮췄다. 기존 기본-head GRPO 대비는 F1 +0.45%p, 생성 토큰 −45.95%다. 따라서 F1 +5%p / 생성 토큰 −30%의 동시 목표는 미달이다. 정확도 향상 또는 비열등성을 입증하지 못했다. 이 검증64는 이미 노출된 개발 자료이며 새로운 최종 성능 확증은 아니다.

## 결과 표

| 조건 | F1 (0–100) | 평균 생성 토큰 | 평균 입력 토큰 | 평균 도구 호출 | 오류 |
|---|---:|---:|---:|---:|---:|
| 선정 head, 추가 GRPO 전 | 25.18 | 76.75 | 3870.83 | 1.7813 | 0 |
| 같은 선정 head, 추가 GRPO 후 | 22.95 | 53.19 | 3973.91 | 1.4219 | 0 |
| 기존 기본-head GRPO | 22.50 | 98.41 | 4625.31 | 2.3281 | 0 |
| Native action generation | 24.70 | 97.44 | 4691.03 | 2.3750 | 0 |
| semantic search 1회 → FINAL | 24.87 | 55.91 | 2220.42 | 1.0000 | 0 |

위 표는 모두 같은 검증64 문항이다. 첫 두 조건은 이번에 함께 평가했다. 아래 세 조건은 앞선 검증64의 저장 결과를 재사용했다. 모델·코퍼스·질문·checkpoint 출처를 검사했으며, 런타임 소스 차이는 선택적 상태 저장 dtype뿐이다. 학습 전 head를 재평가한 64개 모두 기존 결과와 답변·support IDs·F1·생성/입력 토큰·도구 호출·행동 순서가 정확히 일치했다. 재사용 비교의 출처와 검사 결과는 `cross_run_comparison.json`에 보존했다.

| 추가 GRPO 후 − 기준 | F1 변화 (%p) | F1 95% CI (%p) | 생성 토큰 변화 | 생성 비율 95% CI |
|---|---:|---|---:|---|
| 선정 head 학습 전 | −2.23 | [−8.33, +3.24] | −30.70% | [0.5172, 0.8977] |
| 기존 기본-head GRPO | +0.45 | [−7.93, +8.93] | −45.95% | [0.4012, 0.6935] |
| Native | −1.75 | [−10.68, +7.29] | −45.41% | [0.4057, 0.7023] |
| 검색 1회 → FINAL | −1.92 | [−12.74, +8.62] | −4.86% | [0.7134, 1.2232] |

10,000회 paired 문항 bootstrap이다. 개별 지표 marginal 95% CI이며 다중비교 보정·동시 95% 보장·학습 seed 불확실성 추정은 아니다. F1 차이의 구간이 모두 0을 포함한다. 학습 전 대비 입력 토큰은 약 2.66% 늘었으므로 생성 토큰 감소를 전체 계산 비용 또는 지연시간 감소와 동일시하지 않는다. 통제 지연시간 측정은 하지 않았다.

## 무엇을 학습했는가

- 기본 head는 마지막 토큰 hidden을 받는 MLP+skip이다. 앞선 기본-head GRPO는 행동 출력층과 skip만 학습했다.
- 선정 head는 마지막 토큰·질문 끝 토큰·최근 관측 평균 hidden을 연결하는 MLP다. 이전 구조 실험에서 outcome 학습으로 선정한 가중치를 그대로 이어받았다. 초기 checkpoint SHA는 `d987142ff7ea00efe06748fa205e84095a500abc675f290f449e035e70eacb99`다.
- 이번에는 같은 선정 head의 MLP 전체 `net.*`가 업데이트된다. 별도의 GRPO 행동 모델을 붙이거나 BC부터 다시 초기화하지 않았다. 현재 head로 trajectory 수집 → 같은 head 업데이트 → 업데이트된 head로 다음 라운드 수집을 수행했다.
- LLM과 정규화 mean/scale은 고정했다. 시작 head의 고정 복사본은 KL 기준으로만 사용했다. head는 행동 확률을 출력하고 LLM은 인자와 답변을 생성한다.
- train192×4 rollout×2라운드, seed20261055, lr1e-4, 라운드당4epoch, minibatch32, clip.2, KL.02, entropy.01, 보상 F1−.03×cost를 사전 고정했다. 마지막 iteration1을 평가했고 검증 점수로 checkpoint를 고르지 않았다.

기능상 이 모듈은 상태에 조건부인 행동 정책이며, GRPO에서는 최종 결과와 비용으로 학습한다. 하지만 미래를 명시적으로 예측하는 월드모델이나 일반적인 상황 이해 능력이 입증된 decision model은 아니다. 현재 증거는 정책 가중치가 실제 업데이트됐고 개발 평가에서 생성 토큰·도구 호출이 줄었다는 것이다. 단순 검색1회 기준보다 정확도·비용이 우수하다는 증거는 아직 없다.

## 오류와 복구 검증

첫 실행은 768개 trajectory를 수집한 뒤 첫 optimizer update 전에 `behavior logprob mismatch`로 중단됐다. checkpoint와 batch는 생성되지 않았고, optimizer update는0회였다. 원자료768개와 실패 상태를 별도 Drive archive로 보존했다.

평균 hidden은 FP32로 계산되는데 기존 rollout은 학습 저장 시 BF16으로 줄였다. 학습 문항3개의 실제 공개 prefix에서 BF16 변환 시 최대 logprob 차이0.00178051을 재현했고, FP32 유지 시 batch1/32 차이는 최대4.77e-7이었다. 전체 실패 batch는 저장되지 않아 직접 재생했다고 주장하지 않는다.

복구 r1에는 선택적 `state_storage_dtype=torch.float32`를 사용했다. 기존 기본값은 BF16으로 유지해 과거 평가의 실행 동작을 바꾸지 않았다. batch를 확률 검사 전에 저장하도록 수정했다. 관련 회귀/GRPO/rule 테스트6개를 통과했다.

| 검사 | 첫 라운드 | 두 번째 라운드 |
|---|---:|---:|
| 학습 상태 수 | 2,334 | 2,180 |
| GPU replay 최대 logprob 오차 | 9.54e-7 | 9.54e-7 |
| 독립 CPU replay 최대 오차 | 7.15e-7 | 9.54e-7 |
| optimizer update | 292 | 276 |
| 보상 차이가 있는 질문 그룹 | 155/192 | 162/192 |

GPU 검사의 기존 atol/rtol2e-4를 유지했다. 저장 상태는 FP32, 모든 checkpoint 값은 유한값, 정규화는 초기값과 정확히 같고 MLP 가중치4개만 바뀐 것을 확인했다. 실패 실행과 복구 첫 라운드의768개 episode는 답변·support·F1·생성/입력 토큰·도구 호출이 모두 같았다. 수집 행동을 바꿔 검사를 회피한 것이 아니다.

최종 checkpoint SHA는 `6f9e7bbf0f9cb1e158ec38574f8059c93880cb34e00023ae74b5d3b93e43774c`다. 총4,530개 결정 중4,514개가 비강제 학습 상태다. 학습 오류1건·강제 FINAL16건을 제외하지 않았다. 업데이트 후 학습 상태의 action argmax 변화나 clipping 비율은 최적화 진단일 뿐 일반화 증거가 아니다.

## 비용과 보존

| 실행 | Episode | 생성 토큰 | 입력 토큰 |
|---|---:|---:|---:|
| 첫 실패 실행 | 768 | 61,608 | 3,416,046 |
| 복구 r1 학습 | 1,536 | 119,793 | 6,566,548 |
| 학습 전후 검증64 | 128 | 8,316 | 502,063 |
| 합계 | 2,432 | 189,717 | 10,484,657 |

학습 r1은2284.57초, 검증은172.24초였으며 각 단계 상한을 지켰다. CPU 검사·압축·복사와 진단용 train prefix3개 prefill은 위 rollout token 합에 포함하지 않는다. 시간은 공유 환경의 실행 경과로, 통제된 모델별 지연시간 비교가 아니다. 이전 outcome 학습과 초기화 비용도 별도로 이미 발생한 비용이다.

실패 run, 복구 학습 run, 평가 run 모두 각각 archive와 SHA 기록을 남겼다. 복구 학습 archive는1,444,285,309bytes이며 checkpoint와 전체 batch를 포함한다. 원래 최종80과 예비80은 이번 학습·평가에 사용하지 않았다. 앞선 최종평가에서 관측한 정확도 개선 실패를 이번 개발 결과로 덮어쓰지 않는다. 추가 sweep, 월드모델·메모리 확장은 실행하지 않았다.

## 재현 경로

- 계획: `docs/JEV_SELECTED_HEAD_GRPO_CONTINUATION_20261024.md`
- 실행: `scripts/coordinate_decision_selected_head_grpo_r1.py`
- 학습 원자료·checkpoint: `/content/local_fast/lsdrag-text-v1/results/decision_selected_head_grpo192_20261024_r1`
- 전후 평가: `lab/reports/decision_selected_head_grpo_validation64_20261024_r1/`
- 학습 감사: `lab/reports/decision_selected_head_grpo192_20261024_r1/audit.json`
- 실패·원인 진단: `lab/reports/decision_selected_head_grpo192_20261024/replay_diagnosis.json`
- 전체 비용: `lab/reports/decision_campaign_budget_20261023/`
- Git branch: `jev_v1.2`; Drive archive 기록은 각 `lab/registry/decision_selected_head_grpo*/` 아래 보존.

실행 복구·추가 학습·전후 비교는 완료했다. 수치 목표와 새로운 heldout에서의 품질 개선은 달성했다고 주장하지 않는다.
