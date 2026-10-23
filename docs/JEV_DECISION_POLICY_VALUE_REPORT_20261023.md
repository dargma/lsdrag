# 선형 value 대조군 결과 — 2026-10-23

**선형 value는 정확도가 높았지만 효율적 정책 대체재로 채택할 결과는 아니다.** 새60에서 F1.4426,호출5.20,생성157.2,cap27/60,오류5/60. 전체MLPcost는F1.4059,호출2.10,생성59.02,cap/error0. Value의 BC대비 F1차이CI는양수지만 비용이늘었다. 전체정책대비 F1우위는확정되지않았다.

[Protocol](JEV_DECISION_POLICY_VALUE_CONTROL_20261023.md), [전체 수치](../lab/reports/decision_policy_20261023/value_control.json), [PPO·정책수식](JEV_DECISION_POLICY_MATH_20261023.md).

## 새로운 frozen60 결과

| 조건 | F1 | EM | 호출 | 생성token | input | 비용proxy | cap | 오류 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0.3680 | 0.267 | 2.333 | 95.25 | 3962.9 | 3.682 | 0 | 2 |
| historical_bc | 0.3264 | 0.217 | 2.400 | 60.75 | 4026.6 | 3.410 | 0 | 0 |
| full_mlp_cost | 0.4059 | 0.300 | 2.100 | 59.02 | 3725.7 | 3.063 | 0 | 0 |
| stop_only | 0.2680 | 0.183 | 2.100 | 52.85 | 3659.2 | 2.994 | 0 | 0 |
| linear_value | 0.4426 | 0.383 | 5.200 | 157.20 | 12317.2 | 8.087 | 27 | 5 |

## 사전 지정 paired 비교

| 차이 | F1 delta | 95% bootstrap CI | 승/패 | 호출 delta |
|---|---:|---|---|---:|
| linear_value_minus_historical_bc | +0.11624 | [+0.03668, +0.19890] | 16/4 | +2.800 |
| linear_value_minus_full_mlp_cost | +0.03672 | [-0.04394, +0.11890] | 10/7 | +3.100 |
| linear_value_minus_stop_only | +0.17457 | [+0.08906, +0.26222] | 17/3 | +3.100 |

## 배운 점과 한계

단순 회귀의 낮은 train MSE가 종료를 포함한 좋은 전체 trajectory를 보장하지 않는다. 종료불안정·reference Q와 배포정책의 차이·희소상태 coverage는 원인 후보이며, 이번 결과만으로 단일원인을 확정하지 않는다. Ridge나value학습 일반이불가능하다는 결론도아니다.

STOP-only는이번별도60에서F1.2680/calls2.1로BC보다품질이낮았다. 첫STOP실험의평균상승이모든질문군에서재현된것은아니다. 반면fullMLPcost의BC대비평균방향은긍정적이나동일한학습seed/checkpoint의다른문항평가이지독립학습재현은아니다.

원본340평가평균재집계,head결정1076개확률/합법성,checkpoint해시검증통과. Elapsed487.01초(load포함),생성28673,input1913242. 시간이통제된speedup결과는아니다.

학습은같은139상태의두reference target을pooled회귀했다. 다른정책의순차gradient학습과초기화/estimator가달라방법단독인과비교가아니다. 단일수집seed자료이며닫힌형식해를seed만바꿔반복해도독립재현이되지않는다. Actionsoftmax는정답확률이아니다. 새60은후속튜닝에서제외한다.

## 학습 수식

행동 $a$가 합법이었던 train상태 $I_a$,표준화hidden $x_i$,문항균등가중치 $\omega_i$에서

$$
(w_a,b_a)=\arg\min_{w,b}\sum_{i\in I_a}\omega_i(w^\top x_i+b-\widehat Q_i(a))^2+\rho\|w\|_2^2,\qquad\rho=1.
$$

Weighted mean을뺀 $X,Y$에 $\sqrt{\omega}$를곱하면

$$
w_a=X^\top(XX^\top+\rho I)^{-1}Y,\qquad b_a=\bar Q-\bar x^\top w_a.
$$

예측후 $a^*=\arg\max_{a\in\mathcal A(s)}(w_a^\top x+b_a)$를실행한다. Target의reference차이/두suffix표본한계는그대로남는다. [PPO·정책loss와의비교](JEV_DECISION_POLICY_MATH_20261023.md).



재생성: `python scripts/reporting/summarize_decision_policy_value.py`.

[원본·체크포인트·로그 백업 해시](../lab/registry/decision_policy_value_20261023/archive.json).
