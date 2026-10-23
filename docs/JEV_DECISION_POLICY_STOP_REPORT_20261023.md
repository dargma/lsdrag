# 종료 전용 대조군 결과 — 2026-10-23

**전체 정책은 종료 전용보다 평균 F1이 조금 높고 호출은 적었지만, 차이는 확정되지 않았다.** 전체정책 F1.4320/calls2.00, STOP-only .4162/2.25, full−STOP F1 CI[−.0662,+.0973]. Native F1.4675라 Native 대비 품질 우위도 주장할 수 없다. 비용 절감이 종료만의 효과인지 더 검증해야 한다.


[실험 protocol](JEV_DECISION_POLICY_STOP_CONTROL_20261023.md), [PPO·현재 학습·STOPgate 수식](JEV_DECISION_POLICY_MATH_20261023.md), [검증 수치](../lab/reports/decision_policy_20261023/stop_control.json).

## 새로운 frozen60 결과

| 조건 | F1 | EM | 호출/문항 | 생성/문항 | input/문항 | 비용proxy | cap | 오류 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0.4675 | 0.350 | 2.283 | 95.27 | 3873.2 | 3.623 | 0 | 2 |
| historical_bc | 0.3785 | 0.283 | 2.317 | 58.88 | 3883.6 | 3.294 | 0 | 0 |
| full_mlp_cost | 0.4320 | 0.367 | 2.000 | 58.78 | 3656.2 | 2.953 | 0 | 0 |
| stop_only | 0.4162 | 0.350 | 2.250 | 53.73 | 4113.6 | 3.199 | 2 | 0 |

## 사전 지정 paired 비교

| 차이 | F1 delta | 95% bootstrap CI | 승/패 | 호출 delta |
|---|---:|---|---|---:|
| stop_only_minus_historical_bc | +0.03773 | [-0.03385, +0.12038] | 5/4 | -0.067 |
| full_mlp_cost_minus_stop_only | +0.01580 | [-0.06617, +0.09726] | 6/5 | -0.250 |
| full_mlp_cost_minus_historical_bc | +0.05354 | [-0.04378, +0.15573] | 9/5 | -0.317 |

F1 신뢰구간이0을 포함하면 품질 차이를 확정하지 않는다. 이 CI는 질문 bootstrap이며 학습seed 변동을 포함하지 않는다. 작게 관측된 품질 손실을 사후 noninferiority로 이름 붙이지 않는다.

## 검증과 해석 범위

학습은 pilot2 train139상태/990분기의2round 재사용. 초기139상태BC argmax parity와학습후고정도구불변통과. 실행중nonFINAL 도구선택도BC와대조. 새branch수집0이지만원래수집비용이없어진것은아니다. 272평가원본 평균재집계,648head결정확률검사,checkpoint hash검증통과. worker elapsed300.09초(load포함),생성18063/input1057650.

STOP-only는CONT를선택했을때historicalBC의합법nonFINAL argmax를그대로실행한다. 전체MLPcost는pilot2동결checkpoint다. Gate초기화/출력제약이전체정책과달라순수단일요인의인과적ablation이라주장하지않는다. Target은공유reference suffix의Q이며stop-only의on-policyQ가아니다. 단일seed,질문60,참조정책분포및작은학습데이터한계가있다.

새60은이제노출평가데이터이며후속선형value/보상/threshold튜닝에쓰지않는다. 호출감소와전체토큰/시간감소를구분하고,cap강제종료는학습된STOP으로세지않는다. 추가모델/새Qwen교체없음. latency는통제측정이아니다.

재생성: `python scripts/reporting/summarize_decision_policy_stop.py`.

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


[원본·체크포인트·로그 백업 해시](../lab/registry/decision_policy_stop_20261023/archive.json).
