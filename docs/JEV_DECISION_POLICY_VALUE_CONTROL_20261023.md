# 순차 실험 2: 선형 action-value 대조군 — 2026-10-23

사용자 순차 진행 승인에 따른 두 번째 단계. 첫 STOP-only 평가가 끝난 뒤 실행한다. 조건/새test60은 STOP 결과를 읽기 전에 동결했다. 모델과 도구 계약은 유지하며 OLIVIA에서 착안한 **offline value regression**이다. Online LinUCB·oracle tool feedback을 재현한 것은 아니다.

- Pilot2 train24의 두 round139states/990branches를 합쳐 사용. 같은 input features/target 정보지만 full policy의 순차100epoch 업데이트와 ridge의 pooled fitting은 다른 estimator다.
- 각 합법 행동의 `Q = mean(F1−.03·remaining cost)`를 예측하는 선형 ridge. 기존BC mean/scale, 마지막hidden2560,4개 독립회귀,자유intercept. Action별 관측상태 안에서 문항 기여 동일. Ridge penalty1.0 사전고정,validation선택없음. 작은 dual system으로 닫힌형식 해를 구한다.
- 배포는 합법 predicted-Q argmax. 기존 실행기 계약 때문에 기록하는 softmax(Q)는 임의 척도의 선택분포이며 calibratedconfidence가 아니다. 값/확률을 혼동하지 않는다.
- 학습자료가 고정되면 deterministic solve다. seed만 바꿔 동일해를 여러 번 얻는 것을 독립 학습재현으로 세지 않는다. 뒤의 seed단계는 실제 수집 trajectory/reference 변동부터 필요하다.
- 새family-disjoint60을 별도동결,직전STOPtest60과과거100/20/40·다른노출qid/family제외. Native/BC/fullMLPcost/STOPonly/linearvalue의5조건,validation8+test60=340episodes.
- 기존fullMLP와STOP checkpoint그대로; test성적을보고다른조건을고르지않는다. Primary linearvalue−BC/full/STOP의pairedF1 및호출차이. 단일학습자료실현,pairedCI는seed불확실성을포함하지않음.
- 최대1800초(load후),생성400k/input8m,8tools/512output/context16384. 새branch수집0;원래수집비용도명시.

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

[Frozen manifest](../lab/registry/decision_policy_value_20261023/manifest.json). 검사:weighteddual해가직접primal해와일치,불법행동mask,학습coverage부재시fail.

```
PYTHONPATH=src /content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/run_decision_policy_value.py --output /content/local_fast/lsdrag-text-v1/results/decision_policy_value_20261023
```
