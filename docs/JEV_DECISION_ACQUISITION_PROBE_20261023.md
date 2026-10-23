# 중요 상태 사전 예측과 병목 진단 — 2026-10-23

사용자 “오케이”로 진행 승인, “너가 말한다고 너무 좁게만 보지는 말고”를 반영해 acquisition 가설뿐 아니라 target 잡음/reference 변화와 표현 필요성도 함께 점검한다. CPU-only 기존 데이터 분석이며 새 LLM rollout·평가·정책 학습은 없다.

## 중요한 분리

주 예측 검증은 **pilot2 round0의 고정 BC-reference 분기**만 쓴다. Round1 reference는 전체 train outcome으로 학습한 quality policy이므로 그 상태를 그대로 CV에 섞으면 held-out 질문의 미래 결과가 reference 학습을 통해 영향을 준다. Round1은 별도 reference drift 진단에만 사용한다.

24질문의 shared subquestion을 transitive connected component로 묶어6fold를 고정한다. 모든 상태·두 repeat는 같은 fold에 둔다. Ridge 표준화·계수는 fitting fold로만 계산한다. 과거 validation/test outcome은 사용하지 않는다. Frozen BC는 원래train24 teacher label로학습했으므로 importance target의질문분리이지 완전히새질문의end-to-end policy generalization이 아니다. 이전 aggregate audit을 이미 본 자료라는 점도 명시한다.

## 사전 고정 비교

- 주 learner: 마지막hidden2560+metadata의 ridgeρ1.0. 보조: metadata-only ridgeρ1.0. 재튜닝/하이퍼파라미터검색 없음.
- Metadata: turn,합법행동수,BC확률4,entropy,top2확률차. Q/outcome/answer/gold는feature에없음.
- Label: `max mean(F1−.03cost)−mean utility(BC action)`.
- Baseline: 정확예산uniform subset 기댓값,BCentropy,earlyturn. Outcome로고른post-hoc target은진단행이며배포가능baseline이아님.
- 예산: held-out pool의25%/50% branch수,primary25%. 상태비용=합법행동수×2. 모든방법이같은최대도달가능정수예산을정확히사용하도록knapsack과uniform subset inclusion probability를계산한다. **동일token/시간예산은아니다.**
- 주 측정: 한repeat로bestaction을고르고다른repeat에서BC대비utility를계산한cross-repeatgain의선택상태합. 방향을뒤집어평균한다. 실제policy학습후성능은아직측정하지않는다.
- 불확실성: family cluster bootstrap4000,고정OOF예측. fold학습자료가겹치며재학습/수집seed의전체불확실성을포함하지않는다.

## 전진 gate

Primary hidden+metadata가25%예산에서uniform과entropy보다총cross-repeatgain이높고,6fold중4개이상에서uniform보다높으며,uniform대비조건부familybootstrap95%하한>0인경우에만 실제수집비교를설계한다. 통과도연구성공/탐색절감입증이아니다. 실패하면동일fold에맞춰selector를계속변경하지않고잡음/coverage/종료target 진단을검토한다.

## 수식

상태 중요도 label은 $y_i=\max_a\widehat Q_i(a)-\widehat Q_i(a_{BC})$. 회귀는 질문균등가중 ridge이며 정규화도fold train에서만 계산한다. 예산 $B$에서

$$
S^*=\arg\max_S\sum_{i\in S}\widehat y_i,\qquad \sum_{i\in S}2|\mathcal A(s_i)|=B.
$$

$B$는fraction 이하에서실현가능한최대branch수다. Random은위등식을만족하는모든subset에균등하며dynamic programming으로정확한inclusion기댓값을계산한다. 예측기가 Qlabel을 보기 전에 상태를 선택하는지와, 선택 후 측정값을 계산하는 절차를 구분한다.

## 병행 진단

Round0/1에서qid와이미관측된tool name/arguments/result가동일한prefix를찾아bestaction및Q변화를측정한다. reference변화와두suffix의MC잡음이섞이므로인과효과라고하지않는다. metadata-only가hidden보다좋다면고차원head확대의근거가약해진다. 판별력부재가representation문제인지noisytarget문제인지는후속반복표본이필요하다.

[Frozen manifest](../lab/registry/decision_acquisition_probe_20261023/manifest.json),[구현](../scripts/run_decision_acquisition_probe.py). CPU검사3개:family전이연결보존,exact-budget/random확률bruteforce일치,heldout feature변경으로fit/다른예측이바뀌지않음.
