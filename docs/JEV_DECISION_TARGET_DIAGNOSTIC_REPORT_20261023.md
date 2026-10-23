# 학습 데이터 양·상대 이득 target 진단

exploratory repeated-use train-family OOF diagnostic; no confirmatory method selection or PPO efficacy claim

동일ridgeρ1,학습질문5/10/20개(fold별),3개subset순서. 평가질문은고정. 100%에서3seed는같은자료의중복이며독립학습seed증거가아니다.

| fraction | feature | target | gain mean | subset seed range | train MSE | heldout MSE |
|---|---|---|---:|---|---:|---:|
| 0.25 | last | absolute_q | -0.0283 | [-0.0592, 0.0208] | 0.0002 | 0.2318 |
| 0.25 | last | bc_advantage | -0.0244 | [-0.0586, 0.0244] | 0.0002 | 0.1644 |
| 0.25 | metadata | absolute_q | -0.0064 | [-0.0342, 0.0320] | 0.1099 | 3.3887 |
| 0.25 | metadata | bc_advantage | -0.0043 | [-0.0241, 0.0298] | 0.0605 | 1.4937 |
| 0.5 | last | absolute_q | -0.0377 | [-0.0429, -0.0342] | 0.0027 | 0.2052 |
| 0.5 | last | bc_advantage | -0.0365 | [-0.0522, -0.0273] | 0.0019 | 0.1642 |
| 0.5 | metadata | absolute_q | 0.0107 | [-0.0102, 0.0313] | 0.1352 | 0.1546 |
| 0.5 | metadata | bc_advantage | 0.0031 | [-0.0102, 0.0176] | 0.0700 | 0.1006 |
| 1.0 | last | absolute_q | -0.0362 | [-0.0362, -0.0362] | 0.0097 | 0.2123 |
| 1.0 | last | bc_advantage | -0.0581 | [-0.0581, -0.0581] | 0.0080 | 0.1870 |
| 1.0 | metadata | absolute_q | 0.0066 | [0.0066, 0.0066] | 0.1433 | 0.1508 |
| 1.0 | metadata | bc_advantage | 0.0177 | [0.0177, 0.0177] | 0.0847 | 0.1005 |

상대target=Q(s,a)−Q(s,BC(s)). 같은 상태의 행동 순위는 유지하면서 상태별 공통 성분을 제거한다.

상대target은 같은 상태의BC가치를 빼며 정답/후속결과는target으로만사용한다. PPO advantage estimator 자체나 PPO학습의재현이아니다. target별분산이다르므로MSE절대크기를서로비교해우월성을판정하지않는다. train MSE는row-level진단,heldout MSE는질문균등. 작은nestedcurve로더많은데이터가반드시해결한다고추정하지않는다. 모든결과는이미여러번본trainpool의탐색적진단.

[계획](JEV_DECISION_TARGET_DIAGNOSTIC_20261023.md)
