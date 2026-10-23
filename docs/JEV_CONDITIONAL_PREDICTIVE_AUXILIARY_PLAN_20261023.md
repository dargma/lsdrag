# 조건부 후속: 행동 조건부 미래표현 예측 보조학습

2026-10-23 사용자 명시 요청: 아이디어를 저장하고 현재 목표를 달성하면 바로 다음 실험으로 수행. 현재 A-RAG 주실험을 바꾸거나 GPU worker를 병렬기동하는 요청은 아님.

## 실행 조건과 우선순위

현재 개선 head가 미사용 공통 평가셋에서 고정 GRPO96 대비 F1 +0.05(5 percentage points) 이상 및 평균 생성token ratio <=.70를 동시에 달성하면 후속실험 실행을 승인받은 상태. validation32 성과만으로 trigger 금지. point target 달성과 paired CI/seed 재현성은 따로 보고. 현재목표미달이면보류하며성공으로포장하거나조건을완화하지않는다. 현재8시간상한22:48UTC 및 단일GPU 제약 유지;남은예산내에서실행하고충분한시간이없으면정확한재개상태를저장.

## 아이디어

시점 t는 생성token위치가아닌도구실행전후decision step. 공통 frozen LLM hidden h_t를 작은 trainable shared encoder E로 z_t로변환. policy는 z_t로행동확률을출력. 별도작은predictor P는현재z_t와행동a_t(사용가능하면실제로선택된args표현)를받아다음상태표현을예측.

z_t=E_theta(h_t)
hat_z_next=P_phi(z_t,a_t,args_t)
L_pred=||hat_z_next-stopgrad(E_target(h_next))||^2
L=L_policy+beta*L_pred

현재표현과다음표현을직접동일하게만드는temporal smoothing은사용하지않음. 새로운근거가상태를바꾸는차이를보존해야한다. 예측대상은4개행동확률이아닌정책이사용하는상태표현. EMA/고정target,비대칭predictor,표현variance와effective rank진단등collapse대응을실제설계에포함;stop-gradient하나가붕괴방지를보장한다고주장하지않음. 원LLM은고정하고policy와prediction이작은표현모듈을공유해야보조학습이policy표현을바꿀수있음.

## 최소 비교와 저비용 실행

- 같은초기화/구조/자료/업데이트예산의policy-only(beta0) vs policy+prediction(beta>0). 기존선형head에새sharedMLP를추가한다면구조변경대조가필수. 기존챔피언도배포성능reference로포함.
- 우선기존train trajectory의연속hidden/action쌍을재사용. trajectory ID와순서검증,서로다른질문/episode경계pair금지,terminal FINAL에가짜next state생성금지. 실패/forcedcap처리명시. 191state분기cache에는모든suffixhidden이저장되어있지않으므로기존GRPO batch/episode자료에서실제pair복원가능성을먼저확인. 없으면추가추출비용계상.
- 동일train/validation family 규칙,새heldout으로평가. 현재목표달성에사용한test를보조학습튜닝에재사용하지않음.
- action-only조건은인자생성기하의다음표현분포를평균적으로예측하는제약을명시. args를넣을때미래toolresult나정답을input으로넣지않음.
- predictor는학습시만사용가능하므로추론추가연산을제거할수있다. shared encoder의비용은그대로남으며policy-only동일구조와비교해야zero-extra-predictor-cost주장가능.
- beta는loss scale을train에서확인후소수값을validation으로선택하고test전동결. 큰구조/하이퍼전조합탐색금지.

## 판정

주판정은새질문의closed-loop F1/generated/input/calls이며prediction loss감소는보조지표. 예측오차감소만으로policy개선/월드모델획득/추론능력을주장하지않음. 미래tool관측은본질적으로불확실하므로한step latent regression의한계와부분관측성을명시. 성공시에도우선'미래표현예측으로보강한정책표현학습'으로기술. latent rollout으로가상계획을하는큰worldmodel확장은별도범위.

## 근거

- SPR, Data-Efficient Reinforcement Learning with Self-Predictive Representations: https://arxiv.org/abs/2007.05929
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA 2: https://arxiv.org/abs/2506.09985

새알고리즘자체의신규성주장아님. text agent의작은decision head에서효용과비용을입증하는적용가설.
