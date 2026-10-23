# Decision policy pilot2 결과 — 2026-10-23

**비용을 고려한 정책의 평균 품질·비용 동시 개선 신호가 관측됐지만, 통계적으로 확정되지 않았다.** 사전 지정 주조건 MLP cost policy는 BC보다 F1 +0.05856, 도구 호출 −0.5/문항이다. F1 paired95% CI는 [−0.06234,+0.18360]으로0을 포함한다. 단일seed·fresh40 결과이며 일반화/비열등성/논문 novelty 증거로 확대하지 않는다.

[동결 protocol](JEV_DECISION_POLICY_PILOT2_20261023.md), [검증된 전체 수치](../lab/reports/decision_policy_20261023/pilot2.json). Source fd756eb; run `/content/local_fast/lsdrag-text-v1/results/decision_policy_20261023_pilot2`.

## Frozen test40 결과

| 조건 | F1 | EM | 도구/문항 | 생성token/문항 | input/문항 | 비용proxy | cap FINAL | 오류문항 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0.2844 | 0.200 | 2.200 | 92.300 | 3816.2 | 3.505 | 0 | 2 |
| historical_bc | 0.2914 | 0.225 | 2.425 | 59.550 | 4158.1 | 3.436 | 0 | 0 |
| mlp_pg_quality | 0.4183 | 0.325 | 2.725 | 68.600 | 4911.2 | 3.902 | 0 | 0 |
| mlp_pg_cost | 0.3500 | 0.250 | 1.925 | 56.125 | 3543.8 | 2.841 | 0 | 0 |
| mlp_ce_quality | 0.4825 | 0.425 | 6.175 | 142.000 | 14675.6 | 9.088 | 27 | 1 |
| mlp_ce_cost | 0.4483 | 0.375 | 5.125 | 109.525 | 12343.9 | 7.455 | 22 | 0 |
| linear_pg_cost | 0.3725 | 0.275 | 1.900 | 53.975 | 3482.4 | 2.788 | 0 | 0 |
| linear_ce_cost | 0.3183 | 0.275 | 4.425 | 103.575 | 9999.5 | 6.486 | 15 | 1 |

비용proxy는 tool calls + generated/100 + input/10000이다. 토큰 단위별 가중치는 실측 시간·금액이 아니다. supplied prefix 토큰은 별도 기록하며 생성token 절감만으로 전체계산 절감이라고 부르지 않는다. Shared-GPU 시간 관측은 통제 speedup이 아니다.

## 해석

- 사전 주조건 `mlp_pg_cost`: BC 대비 호출 약20.6%, 생성token 약5.8%, input 약14.8%, 비용proxy 약17.3% 감소. F1/EM 평균은 상승했으나 정답 안정성은 아직 불확실하다. 오류·cap 종료는0이다.
- `mlp_pg_quality`: BC보다 F1이 높지만 호출도 증가한다. 같은 목적에서 cost항 추가는 평균 호출을 낮추는 대신 F1도 낮췄다. 보상 trade-off가 실제 행동에 반영됐다는 관찰이지 모든 질문에서 Pareto 우세라는 뜻은 아니다.
- outcome CE도 미래 결과 학습이다. MLP CE는 더 높은 F1을 얻었으나 호출5.125~6.175회와 cap 종료22~27/40으로 비용이 컸다. PG가 CE보다 정확도도 우월하다는 결론은 틀리다.
- `linear_pg_cost`는 F1.3725/calls1.9로 유망한 사전 포함 보조조건이다. 이 test를 보고 최적 모델로 선택해 재평가 점수처럼 보고하지 않는다. 복잡한 head가 필요하다는 증거는 없다.

## 실행·검증과 한계

전체1422 episodes: collectors48 + branches990 + val64 + test320. 학습2round, 총1367.39초(load포함), 생성79290 / input5976627 tokens. 두 round의 상태수는67/72. 여섯 learner는 round별 동일 데이터 SHA를 사용했다.
원본 평가384개 평균 재집계, head결정1534개 확률 합·합법 행동, frozen checkpoint/data hash 검증 통과. final status complete. 재생성: `python scripts/reporting/summarize_decision_policy_followup.py`.

본 방법은 shared-reference Monte Carlo return 기반 근사 정책 개선으로, PPO/불편향 on-policy gradient가 아니다. reference는 round0 tempered BC, round1 qualityPG이며 모든 learner에 중립적인 상태분포는 아니다. 두 suffix sample만으로 Q순위가 안정적이라고 보장하지 않는다. Backbone·생성기·분기는 고정; 추가 policy backbone 없음. 행동확률은 calibration된 성공확률이 아니다.

원래test100과 pilot1test20에 더해 **이번40도 노출된 평가 데이터**다. 후속 학습·보상/threshold/구조 튜닝에 쓰지 않는다. 후속 후보는 related-work 보고서의 종료전용/선형value/강한최적화Native 대조군이며, 현재 결과에 새 조건을 사후추가하지 않았다.

원본·체크포인트·로그는 Drive에 묶어 백업하고 SHA256을 검증했다. [archive provenance](../lab/registry/decision_policy_pilot2_20261023/archive.json).

## 학습 수식

[PPO·return/advantage·반사실 Q·outcome CE의 수식과 코드 대응](JEV_DECISION_POLICY_MATH_20261023.md)을 함께 읽는다. Pilot1 PPO와 pilot2 expected-return 학습은 다른 알고리즘이다.

## 4. Pilot2: 반사실 action return의 근사 정책 개선

같은 상태에서 각 합법 행동을 실행하고, 이후는 고정 reference $\pi_k$로 끝까지 진행한다. 행동당 $M=2$ suffix 표본으로

$$
\widehat Q_k(s,a)=\frac1M\sum_{j=1}^{M}\left[F(\tau_{s,a,j}^{\pi_k})-\lambda C(\tau_{s,a,j}^{\pi_k})\right].
$$

Cost 조건은 $\lambda=0.03$, quality-only 조건은0. Round0 reference는 temperature8 BC, round1은 round0 MLP quality policy다. 여섯 learner는 같은 표본을 받는다. 이 값은 **reference continuation 아래의 Q 추정치**이며 최적 $Q^*$나 모든 learner의 현재 Q가 아니다.

Question $q_i$에서 나온 상태 수를 $n_{q_i}$라 두고 $w_i=(1/n_{q_i})/\sum_j(1/n_{q_j})$로 가중한다. 각 문항의 총 기여를 같게 만든다. $b_i$는 합법 행동 Q의 평균이다.

$$
\mathcal L_{\mathrm{return}}=-\sum_iw_i\sum_a\pi_\theta(a|s_i)(\widehat Q_k(s_i,a)-b_i)
+\beta\sum_iw_iD_{KL}(\pi_\theta\Vert\pi_0)
-\alpha\sum_iw_iH(\pi_\theta),
$$

$$
\beta=0.02,\qquad\alpha=0.01.
$$

$Q,b,\pi_0$는 학습 중 고정한다. 상태별 상수 $b_i$는 이 full-action 합의 정확한 gradient를 바꾸지 않는다. 초기 $\pi_0$는 temperature8 BC이며 greedy 초기 선택은 보존된다. 이 loss에는 PPO probability ratio/clipping이 없다. **현재 보고서의 PG는 all-action expected-return surrogate이며 PPO·불편향 on-policy gradient가 아니다.** [구현 improvement_loss/fit_improvement](../src/lsdrag/decision_policy_followup.py). 비용민감 정책 학습과 all-action gradient의 관련 선행은 [AggreVaTeD](https://arxiv.org/html/1703.01030v1)지만, expert oracle 가정을 우리 noisy reference rollout에 대입하지 않는다.

## 5. 동일 outcome 데이터의 soft CE

$$
y_i(a)=\operatorname{softmax}_{a\in\mathcal A(s_i)}\left(\frac{\widehat Q_k(s_i,a)}{\tau_Q}\right),\qquad\tau_Q=0.1,
$$

$$
\mathcal L_{\mathrm{CE}}=-\sum_iw_i\sum_a y_i(a)\log\pi_\theta(a|s_i)
+\beta\sum_iw_iD_{KL}(\pi_\theta\Vert\pi_0)
-\alpha\sum_iw_iH(\pi_\theta).
$$

이는 teacher가 고른 행동 한 개를 맞추는 BC와 다르다. **Outcome CE도 미래 결과를 이용해 decision policy를 학습한다.** Return loss와 CE는 같은 정보의 다른 학습법이다. KL/entropy 및 target temperature까지 달라질 수 있으므로 알고리즘명만으로 우열을 정할 수 없다.
