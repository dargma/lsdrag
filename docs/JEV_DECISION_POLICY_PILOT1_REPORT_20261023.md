# Decision policy v1.2 pilot1 결과

실제 상태: `complete`. raw: `/content/local_fast/lsdrag-text-v1/results/decision_policy_20261023_pilot1`.

## 입력·구조 진단

Teacher 행동 모방 지표이며 QA 성능이 아니다. train24/validation8, validation29상태.

| 조건 | macro F1 | 조기 FINAL 오분류 | 파라미터(보조 출력 포함) |
|---|---:|---:|---:|
| last_linear | 0.8140 | 0 | 33,293 |
| question_linear | 0.2742 | 12 | 33,293 |
| observation_linear | 0.7474 | 2 | 33,293 |
| last16_linear | 0.7034 | 2 | 33,293 |
| last_mlp | 0.8140 | 0 | 339,729 |
| observation_mean_mlp | 0.6579 | 3 | 339,729 |
| attention32 | 0.6918 | 3 | 330,634 |

사전 규칙으로 선택: `last_mlp`.

## 실제 학습 및 수집

분기 episode 316, train baseline 24, 완료 PPO 업데이트 3.
- PPO 1: 48 episodes, 159 학습상태, 수집 중 F1 0.2588. 이는 on-policy train 성능이다.
- PPO 2: 48 episodes, 158 학습상태, 수집 중 F1 0.2676. 이는 on-policy train 성능이다.
- PPO 3: 48 episodes, 159 학습상태, 수집 중 F1 0.2468. 이는 on-policy train 성능이다.

## 동결 fresh20 평가

| 정책 | F1 | EM | tool calls | 생성 tokens | 전체초(관측) | 오류 | cap 강제종료 |
|---|---:|---:|---:|---:|---:|---:|---:|
| native | 0.2295 | 0.100 | 2.30 | 96.00 | 1.355 | 0 | 0 |
| native_fast | 0.1867 | 0.100 | 2.35 | 64.20 | 1.108 | 0 | 0 |
| historical_bc | 0.2762 | 0.150 | 2.30 | 58.35 | 0.966 | 0 | 0 |
| bc | 0.2762 | 0.150 | 2.30 | 58.35 | 0.965 | 0 | 0 |
| outcome_ce | 0.3724 | 0.250 | 5.90 | 139.15 | 2.501 | 1 | 13 |
| ppo | 0.2762 | 0.150 | 2.30 | 59.70 | 0.985 | 0 | 0 |

질문 단위 paired bootstrap(탐색):
- ppo_minus_bc: F1차이 +0.0000, 95% [+0.0000, +0.0000], 0승/0패.
- outcome_ce_minus_bc: F1차이 +0.0962, 95% [-0.1121, +0.3000], 5승/3패.
- native_fast_minus_bc: F1차이 -0.0895, 95% [-0.2290, +0.0067], 1승/2패.

## 결론과 다음 진단

이번 pilot에서는 PPO 정책 개선·탐색 단계 절감이 입증되지 않았다. F1과EM은 BC와 같고 평균 호출수도 같다. Actor 파라미터는 실제 변경됐으나 fixed train82에서 greedy 행동변경0, validation29에서1개였다. 학습 미실행과 정책 행동변화 부족을 구분한다.

Outcome CE는 fresh20에서 F1 점추정이 높지만 paired CI가0을 포함하며, 평균5.9호출/13개cap 강제종료/1개생성truncation으로 종료 제어가 나쁘다. 정확도·효율 공동 개선으로 채택하지 않는다.

다음 후보는 train-only에서 PPO logit margin/advantage/KL 진단, 후기상태·종료행동에 대한 반사실 coverage 보강, 동일 수집데이터 예산의 outcome CE 대 PPO 비교다. 이번 새20은 이제 노출되었으므로 후속튜닝금지. 추가실험은 아직 실행하지 않았다.

## 해석 제한

- Single-seed small pilot; final20 cannot be reused for tuning.
- Branch states are exposed train24; no heldout generalization from branch gains.
- Success/cost auxiliary heads are not calibrated current-PPO success probabilities.
- Native-fast raw policy_probs are uniform placeholders, not native confidence; excluded from probability analysis. Head policy probabilities are actual outputs.
- Observed wall times include diagnostic readout/CPU-copy overhead; no controlled speedup claim.
- Attention reads32 context-bin means, not full-token attention.
- Native-fast changes prefix constraints; not an exact native decoder replacement.
- Outcome CE uses branch data; PPO additionally collects144 on-policy episodes. Any difference is not an equal-data algorithm-only comparison.

원자료의 실패와 timeout도 포함한다. 구조 개선, RL 개선, 시간 개선과 신규성은 별개로 판단한다.
