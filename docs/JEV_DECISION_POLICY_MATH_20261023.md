# Decision policy 수식과 실제 구현

이 문서는 pilot1의 PPO와 pilot2의 반사실 정책 개선을 구분한다. 아래 수식은 현재 코드의 objective를 설명하며, 성능 보장을 뜻하지 않는다.

## 1. 상태·행동·head

상태 $s_t$는 질문과 현재까지 관측한 도구 결과다. Frozen LLM의 현재 prefill 표현을 $h_t=f_\phi(s_t)$, 합법 행동 집합을 $\mathcal A(s_t)$라 두면

$$
\pi_\theta(a\mid s_t)=\frac{\exp z_\theta(h_t)_a}{\sum_{b\in\mathcal A(s_t)}\exp z_\theta(h_t)_b},\qquad a\in\mathcal A(s_t).
$$

불법 행동 확률은0이다. $\phi$는 고정, head의 $\theta$만 학습한다. 행동은 keyword_search/semantic_search/read_chunk/FINAL이고, 선택 뒤 같은 LLM이 인자·답변을 생성한다. 배포는 greedy argmax이며 trajectory 수집은 지정된 stochastic policy를 사용한다. $\pi(a|s)$는 행동 선택 확률이지 답변 정답확률이 아니다.

## 2. 품질·비용·return

현재 비용 proxy는

$$
c_t=\mathbf1[a_t\ne\mathrm{FINAL}]+\frac{n_t^{\mathrm{generated}}}{100}+\frac{n_t^{\mathrm{input}}}{10000}.
$$

Input에는 실제 처리한 prefill과 공급 prefix가 포함된다. 최종 답변 F1을 $F(\tau)$라 두고 discount $\gamma=1$로

$$
G_t=F(\tau)-\lambda\sum_{u=t}^{T}c_u.
$$

FINAL에서도 답변 생성 비용은 존재한다. 이 비용은 시간·금액 실측값이 아니다. 정확도와 호출/토큰을 별도 보고해야 한다. 구현: [returns_for_episode·rollout](../src/lsdrag/decision_policy.py).

## 3. Pilot1: 실제 실행한 PPO

[PPO 원 논문](https://arxiv.org/abs/1707.06347)의 clipped surrogate를 사용한다. Pilot1의 수집 정책은 합법 행동 uniform exploration을 섞는다.

$$
\widetilde\pi_\theta=(1-\epsilon)\pi_\theta+\epsilon U_{\mathcal A},\qquad\epsilon=0.05.
$$

Old behavior probability를 저장하고 새 policy에도 같은 mixture를 사용한다.

$$
r_t(\theta)=\frac{\widetilde\pi_\theta(a_t|s_t)}{\widetilde\pi_{\mathrm{old}}(a_t|s_t)},\qquad
A_t=G_t-V_{\mathrm{old}}(s_t).
$$

Batch 내 $A_t$를 평균0/표준편차1로 표준화한 $\widehat A_t$를 고정하여 업데이트한다. **이 구현은 GAE가 아니라 Monte Carlo return-minus-value**다.

$$
\mathcal L_{\mathrm{clip}}=-\mathbb E_t\left[\min\left(r_t\widehat A_t,\operatorname{clip}(r_t,1-\delta,1+\delta)\widehat A_t\right)\right],\qquad\delta=0.2.
$$

코드에서 최소화한 총 loss는

$$
\mathcal L_{\mathrm{PPO}}=\mathcal L_{\mathrm{clip}}+0.5\mathbb E_t(V_\theta-G_t)^2+0.02\mathbb E_tD_{KL}(\widetilde\pi_\theta\Vert\widetilde\pi_{BC})-0.01\mathbb E_tH(\widetilde\pi_\theta).
$$

Critic return target과 advantage는 rollout 후 고정된다. Cap 강제 FINAL은 actor 학습 batch에서 제외하지만 앞선 step의 return에는 그 비용·결과가 반영된다. 반사실 개입 branch는 on-policy PPO batch에 넣지 않는다. 3회 수집/업데이트, 각4epoch, minibatch32. Pilot1 비용계수는 $\lambda=0.005$이며 초기화는 pilot1 manifest를 따른다. 구현: [clipped_policy_loss](../src/lsdrag/decision_policy.py), [pilot1 PPO loop](../scripts/run_decision_policy_pilot.py).

### 왜 탐색했는데 행동이 잘 안 바뀔 수 있는가?

Mixture에서 uniform 항은 $\theta$와 독립이므로

$$
\nabla_\theta\log\widetilde\pi_\theta(a|s)=\frac{(1-\epsilon)\pi_\theta(a|s)}{\widetilde\pi_\theta(a|s)}\nabla_\theta\log\pi_\theta(a|s).
$$

Actor가 거의0 확률을 준 대안을 uniform이 뽑으면 앞의 계수가 작다. 이는 pilot1 train-only 포화 진단과 연결되는 기전이며, 성능 부진의 유일한 원인이라고 증명한 것은 아니다. [진단 코드](../scripts/reporting/audit_policy_train.py).

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

## 6. 다음 단계: 종료 전용 대조군

BC의 합법 non-FINAL 도구를

$$
a_{BC}^{tool}(s)=\arg\max_{a\in\mathcal A(s)\setminus\{\mathrm{FINAL}\}}z_{BC}(s)_a
$$

로 고정한다. Binary gate $g_\psi(s)$는 STOP 또는 CONTINUE만 고르고, CONTINUE면 반드시 $a_{BC}^{tool}(s)$를 실행한다. Shared-reference target은

$$
Q_{stop}(s)=\widehat Q(s,\mathrm{FINAL}),\qquad Q_{cont}(s)=\widehat Q(s,a_{BC}^{tool}(s)).
$$

같은 return objective를 두 선택에 적용한다. CONT 이후의 target suffix가 reference policy라는 사실을 보존하며, 전체를 stop-only 정책으로 실행한 새 episode 평가로 실제 효과를 확인한다. 이 baseline은 Stop-RAG에서 착안한 변형이며 원 논문의 full-horizon Q(λ)/별도 encoder를 재현한 것은 아니다. 강제 cap 종료는 학습된 STOP 성공으로 계산하지 않는다.
