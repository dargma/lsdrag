# 고정 reference 반복 신뢰성 결과

12개 학습 상태, 합법행동당 새6회, 총252회. 정책 학습이나 held-out QA 평가가 아닌 train-only 진단.

| 지표 | 결과 |
|---|---:|
| cross_half_gain_mean | 0.18424416203703706 |
| question_bootstrap_ci95 | [0.04300209340277779, 0.3475417636574075] |
| optimistic_gain_mean | 0.18424416203703706 |
| half_argmax_agreement | 1.0 |
| half_selected_within_005_both | 1.0 |
| old_new_best_agreement | 0.9166666666666666 |
| positive_cross_half_states | 6 |
| negative_cross_half_states | 0 |

Cross-half: 앞3회로 선택한 행동을 뒤3회에서 BC와 비교하고 역방향도 평균. Utility=F1−.03cost. CI는12질문 재표집이며 고정된 reference/상태에 조건부이다.

학습 신호가 양수라는 제한적 근거가 있다.

기존2회와 새6회 순위 비교는 보조 진단. Greedy FINAL 반복 일치는 독립 성공확률 증거가 아니다. 정책의 일반화, 독립 수집/학습 seed 재현, PPO/GRPO 우월성을 입증하지 않는다.

[사전 protocol](JEV_DECISION_RELIABILITY_20261023.md) · [상태별 수치](../lab/reports/decision_reliability_20261023/result.json)
