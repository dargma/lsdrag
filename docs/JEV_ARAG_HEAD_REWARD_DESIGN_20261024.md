# ARAG 다단계 선택 헤드의 reward·credit assignment 설계안

2026-10-24 사용자 요청: ARAG 멀티턴과 선택 헤드 특성을 고려한 아이디어를 우선 저장. **설계안이며 새 reward 학습 실행 승인이 필요한 것으로 해석하지 않되, 이번 요청에서는 새 실험을 시작하지 않는다.** 진행 중 context-pareto 평가는 변경 없이 마무리한다. 후속 실행 시 별도 프로토콜·예산·입력 해시를 먼저 고정한다.

## 1. 목표와 확인된 구현

고정 backbone Qwen3-4B-Instruct-2507, 4행동(keyword_search, semantic_search, read_chunk, FINAL), 작은 헤드만 학습. 검색 인자·답변은 같은 고정 LLM이 생성한다. 목표는 고정 basicGRPO 대비 F1 +5%p **및** 생성 토큰 -30%; 검증64는 개발용이며 독립 확인을 대체하지 않는다. train192/공유101962코퍼스와 보호된 test 분리를 유지한다.

현재 구현은
\[
R(\tau)=F1(\tau)-0.03\left[N_{tool}+T_{gen}/100+T_{input}/10000\right].
\]
생성100토큰은 reward0.03, 즉 F1 3%p와 교환된다. 하지만 도구 횟수·입력 토큰까지 섞은 scalar reward가 생성 토큰 예산을 보장하지 않는다. 모두 F1=0인4개 rollout 그룹에서는 actor advantage를0으로 두지만 KL/entropy는 남는다.

**코드 확인:** `scripts/run_decision_quality_head_grpo.py`의 actor는 같은 질문 그룹의 최종 episode_reward를 표준화한 advantage를 해당 trajectory 모든 행동에 반복한다. `returns_for_episode`의 단계별 return은 계산·저장되지만 actor loss에는 쓰이지 않는다. `roww`는 trajectory당 가중치를 균등화하므로 단순히 긴 trajectory가 더 많은 행을 갖는 문제는 완화한다. 따라서 문제를 '긴 trajectory가 무조건 더 큰 가중치를 받는다'고 설명하면 부정확하다. 남는 문제는 단계별 성공/낭비의 책임 분리가 없다는 점이다.

관측: qualityMLP32.45/139.03 → GRPO34.75/150.91. 경험적 품질·비용 분리 학습28.17/104.80. 앞선 오프라인 예산 제약도 실제 품질 유지에 실패했다. 이는 목표 불일치·credit assignment·오프폴리시 오차의 가설을 지지하지만 하나의 원인으로 확정하지 않는다.

## 2. ARAG에 맞는 원칙

- 검색은 답을 바로 맞추기보다 다음 검색어/문서 위치를 제공할 수 있다. 읽기는 관측된 snippet을 전체 근거로 바꾼다. 당장의 답변 F1 증가가0이라는 이유만으로 벌하지 않는다.
- FINAL은 '현재 근거로 생성한 답의 기대 품질'과 '추가 정보 획득의 기대 이득 및 비용'을 비교하는 행동이다. 횟수·시간만으로 종료 보상을 주지 않는다.
- 작은 헤드가 정하는 것은 행동이고, 구체적 검색어·답변 생성은 frozen LLM이다. 행동별 평균 성과에는 생성기 오류와 검색 환경의 변동도 포함된다. 헤드에 모든 실패 책임을 확정적으로 주지 않는다.
- 최대 softmax는 행동 선호의 확률이지 정답 확률/epistemic uncertainty가 아니다. 낮은 entropy, 큰 max probability 자체를 보상하면 과신을 학습할 수 있다.

## 3. 권장 주목적: 품질 제약 아래 실제 생성 비용 최소화

목표를 직접 나타내면
\[
\min_\theta\;\mathbb E_\pi[T_{gen}],\quad
\mathbb E_\pi[F1]\ge q_\star,
\]
여기서 최종 성공은 별도로 \(\mathbb E[T_{gen}]\le0.7T_{baseline}\)까지 확인한다. 학습의 \(q_\star\)는 **train에서** 고정한 기준 정책의 paired 성과와 목표 차이를 이용해 정하며, validation 성적을 minibatch reward나 dual 업데이트에 넣지 않는다. feasibility가 확인되지 않은 목표를 보장한다고 말하지 않는다.

실용적 라그랑주안:
\[
L(\theta,\lambda)=\mathbb E[T_{gen}/B]+\lambda(q_\star-\mathbb E[F1]),\quad
\lambda\leftarrow[\lambda+\eta(q_\star-\widehat{F1}_{train})]_+.
\]
\(B\)는 train 기준 정책에서 고정한 토큰 스케일이다. rollout batch의 품질 추정은 지수평활·상한을 사용해 dual 폭주를 방지한다. 이는 고정 penalty보다 목표를 명시하지만 **보장도, 자동 해결책도 아니다**. 이전 제약 실험과 차이는 현재 정책의 새 rollout으로 추정하며, 아래 단계별 credit을 적용한다는 데 있어야 한다. 오래된 suffix cache에서 예산만 다시 조절하는 실험은 반복하지 않는다.

도구 호출 수/입력 토큰은 별도 보고한다. 생성토큰 주목적에 임의로 섞지 않는다. 오류·최대8호출 등 환경 제약은 동일하게 유지하고 실패를 기록한다. latency 최적화는 별도 목표가 승인될 때만 추가한다.

## 4. 가장 중요한 변경: 상태·행동별 미래 성과와 비용

현재 정책을 따라 행동한 뒤 최종 답까지의 기대 결과를 분리한다:
\[
Q_Q(s,a)=\mathbb E[F1_{final}\mid s,a,\pi],\qquad
Q_C(s,a)=\mathbb E[\sum_{u=t}^{T}g_u\mid s,a,\pi],
\]
\(g_u\)는 해당 단계에서 실제 생성한 토큰 수이며 FINAL 답변 토큰도 포함한다. 이미 쓴 토큰은 현재 선택의 비용 advantage에 다시 넣지 않는다. 초기 실험은 \(\gamma=1\): 단지 늦게 얻는 정답이라는 이유로 가치를 깎지 않는다.

4행동의 작은 action space를 이용해
\[
A_Q(s,a)=Q_Q(s,a)-\sum_b\pi_{old}(b\mid s)Q_Q(s,b),
\quad
A_C(s,a)=Q_C(s,a)-\sum_b\pi_{old}(b\mid s)Q_C(s,b),
\]
\[
A(s,a)=\lambda A_Q(s,a)-A_C(s,a)/B.
\]
학습된 작은 quality/cost 출력을 critic으로 활용할 수 있다. critic은 train-only Monte Carlo/TD target으로 갱신하고 actor target은 detach한다. target head/시차 업데이트로 자기 예측을 즉시 정답 취급하는 순환을 완화한다. 배포에서는 actor만 사용해 추가 LLM/judge 호출을 없앨 수 있다.

헤드 확률의 실질적 사용은 (1) 위 기대 baseline, (2) PPO/GRPO likelihood ratio, (3) reference KL, (4) 중요 상태의 추가 rollout 배분이다. 단순히 확률을 reward로 주는 방식이 아니다. 합법행동 mask와 강제FINAL의 비학습 처리를 일치시킨다.

### 실행 가능한 두 경로

**저비용 첫 단계:** 기존 on-policy rollout의 단계별 remaining generated를 사용하고, 최종F1용 state baseline과 remaining cost baseline을 분리한다. 작은 critic의 일반화가 나쁘면 질문 그룹 baseline을 유지하되 남은 비용의 step credit부터 고친다. 이것만으로 정확한 action credit이 해결된다고 주장하지 않는다.

**핵심 강화 단계:** 동일 prefix에서 FINAL과 선택된 continue행동을 분기하여 current-policy suffix를 샘플링한다. 모든 행동×모든 상태를 수집하기보다 actor의 FINAL/continue가 경쟁하고 결과 추정이 불확실한 train 상태에 집중한다. 분기 자료는 critic/오프라인 선호학습에 사용하며 on-policy PPO batch로 가장하지 않는다. 다른 행동 사이의 난수 seed 공유만으로 통계적으로 잘 통제된 common random numbers가 된다고 가정하지 않는다.

## 5. 종료의 기회비용

\[
\Delta_Q(s)=\max_{a\ne FINAL}Q_Q(s,a)-Q_Q(s,FINAL),
\]
\[
\Delta_U(s,a)=\lambda[Q_Q(s,a)-Q_Q(s,FINAL)]
-[Q_C(s,a)-Q_C(s,FINAL)]/B.
\]
종료 여부는 이러한 비교를 배우게 한다. min/max 두 샘플의 경험적 분리도 일반화 보장은 아니며, 이전 Pareto 결과가 실제 실패 사례다. 서로 엇갈리는 rollout에서는 hard label을 피하고 reference KL/낮은 신호 가중치/추가 표본을 택한다. '근거 있음'이나 '확신 높음'을 곧바로 정답 보상으로 대체하지 않는다.

위 opportunity target은 단계별 advantage의 추정/보조학습에 사용한다. terminalF1와 같은 근거를 중복 reward로 더해 실제 목적을 바꾸지 않는다. 모든 실패 그룹에서 비용 최적화만 남는 것을 막되, 모든 학습까지영구중단하지 않는다: 품질신호부족이면 성공 경로 탐색과 critic 데이터 보강을 우선하며 zero-cost 종료를 성공으로 처리하지 않는다.

## 6. 보조 신호: 조건부로만 사용

- 반복 검색/읽기: 동일인자뿐 아니라 반환근거와 지식상태가 변했는지 확인. 동일행동 일괄 벌점은 유용한 재검색도 막는다. 추가 비용은 이미 실제토큰에 반영되므로 중복 벌점을 기본값으로 넣지 않는다.
- 근거 coverage·새 문서·bridge entity: train의 진단/critic 보조표적으로 검토 가능. 검색결과 개수나 novelty만 보상하면 불필요한 검색을 늘릴 수 있다. goldsupport는 train label에서만 접근하며 입력·test 환경으로 유출하지 않는다.
- potential shaping을 쓴다면 \(r'_t=r_t+\gamma\Phi(s_{t+1})-\Phi(s_t)\), terminal potential0 등 telescoping 조건을 명시한다. 학습 중 바뀌는 근사potential의 정책불변성은 자동으로 성립하지 않는다. 첫 파일럿에서는 제외해 원인 해석을 유지한다.
- 행동별 고정 reward(semantic +, read - 등), 고정step수 보상, confidence 보상은 채택하지 않는다.

## 7. 효율적인 후속 순서와 판정

이번 요청은 저장 단계이며 새 reward run 미실행. 진행 중 구조평가가 끝나면:
1. 같은 초기checkpoint·train192·seed·rollout수로 현재 reward/actor 구현과 단계별 품질·생성비용 분리 방식을 비교. 기존결과를 비교기준으로 쓸 경우 조건차이와 노출을 명시한다.
2. 첫 실행부터 여러 shaping을 섞지 않는다. 필요 시에만 현재정책의 소수prefix분기 추가. 단순 계산예산 증가와 알고리즘 효과를 구분한다.
3. validation64에서 F1·실제generated·입력·호출·zero-tool·강제FINAL·오류를 짝지어 보고. F1보존 실패면 제약 충족으로 포장하지 않는다. 학습critic 오차 감소/entropy감소는 정책효과 증거가 아니다.
4. 고정한 별도 독립평가에서 공동수치목표와 상태조건부정책 증거까지 확인해야 목표완료. 기존final80실패와 reserved80보호 유지.

이 문서는 구현 코드와 프로젝트 관측을 토대로 한 설계 제안이며, 기존 RL 방법의 결합 자체를 연구 신규성으로 주장하지 않는다.
