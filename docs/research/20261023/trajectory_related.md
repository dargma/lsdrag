# Trajectory·반사실 결과 기반 정책 학습 관련 연구

검토일·검색 상한: 2026-10-23. 최신성을 검색한 뒤 논문 원문 방법·부록을 확인한 선택적 검토다. 발표 venue나 저자 수치의 재현을 의미하지 않는다. 각 원문의 검토 범위와 미확인 항목은 [source ledger](../../../lab/reports/related_work_20261023/trajectory_sources.json)에 기록했다. 실행 중인 [pilot2 protocol](../../JEV_DECISION_POLICY_PILOT2_20261023.md)은 바꾸지 않았다.

## 결론

**행동 이후의 최종 결과를 비교해 작은 정책을 학습한다는 원리 자체는 새롭지 않다.** 우리 후보 contribution은 이미 계산된 frozen 생성 모델의 hidden state에서 다종 도구·종료를 직접 제어하고, 같은 모델로 인자·답변을 계속 생성하는 구현에서 **추가 모델 없이 품질–비용 개선을 얼마나 작은 학습 예산으로 얻는가**에 있다. 이는 연구 가설이며 현재 성능·독창성 입증이 아니다.

## 가까운 원문 다섯 편

| 연구·확인 버전 | 방법과 학습 대상 | 데이터·우리 실험과의 관계 |
|---|---|---|
| [CIPO](https://arxiv.org/html/2610.04991v1), 2026-10-04 v1 | 같은 prefix에서 atomic tool ↔ composite skill 대안을 실행하고 현재 정책으로 끝까지 진행. 결과 차이를 GRPO 보조 보상에 반영. LLM LoRA 학습. | TOOLATHLON·TOUCAN·TRAJECT-Bench. 효율적인 도구 결정을 반사실 rollout으로 학습한다는 점이 직접 겹친다. 우리는 skill 생성 없이 고정 4행동 head만 학습한다. |
| [CRR](https://arxiv.org/html/2609.33875v3), 최초 2026-09-27, v3 10-03 | 실행 환경을 복원해 선택된 행동의 대안을 시도하고 terminal-return 차이를 해당 step advantage로 사용. Qwen3-14B LoRA. | 결정적 실행 조건을 통과한 SWE-Gym·SWE-rebench 3,638개. 우리와 달리 코드 환경·생성 정책 학습이다. replay 비용을 포함한 비교가 핵심 참고점이다. |
| [FORGE](https://arxiv.org/html/2609.34358v1), 2026-09-28 v1 | frozen host 주변의 269K router가 증거 형식·thinking 예산 선택. 후보 utility 열거→Boltzmann KL 증류→group-relative refinement. | HotpotQA·2Wiki·MuSiQue·PopQA·FEVER. frozen LLM+작은 router+결과/비용 학습의 강한 선행. 다회차 도구 정책 대신 질문별 routing이다. |
| [AggreVaTe / NRPI](https://arxiv.org/html/1406.5979v1), 2014-06-23 v1 | expert 또는 현재 learner의 continuation cost-to-go를 수집해 cost-sensitive 정책 학습. | 일반 순차결정 알고리즘이다. 우리처럼 reference 정책으로 suffix를 실행하는 방식은 단순 teacher-action 모방보다 approximate policy iteration 계보에 가깝다. |
| [AggreVaTeD](https://arxiv.org/html/1703.01030v1), 2017-03-03 v1 | discrete action 전체의 정책확률×expert cost-to-go를 미분하는 neural policy 학습. | 로봇·dependency parsing. 현재 all-action expected-return objective의 기본 gradient 형태는 선행한다. 우리에게 near-optimal expert oracle 보장은 없다. |

추가 건축적 선행인 **LAnR은 2026-10-19 v3가 최신**이다. 기존 저장 메모의 v2와 구별해야 한다. 같은 hidden state를 이용한 retrieval/stop head는 [최신 원문](https://arxiv.org/html/2604.17866v3)을 기준으로 head 관련 검토와 함께 읽는다.

## 비교에서 놓치면 안 되는 조건

**CIPO:** SFT는 성공한 augmented-action trajectory를 사용하며, policy 단계는 고정 4-rollout group 안에 base·branch를 배치한다. 정확한 training-instance 수는 확인하지 못했다. atomic-only GRPO는 action space와 SFT 초기화도 달라, 반사실 신호의 효과를 보려면 GRPO+Skills와 비교해야 한다. 주 TSR의 반복 judge 평가는 독립 training seed 반복이 아니다. SFT rank64 / policy rank32 LoRA로 생성 모델 내부를 업데이트하므로 head-only와 학습 비용이 다르다. [방법·부록 I/O](https://arxiv.org/html/2610.04991v1)

**CRR:** 복원 가능한 환경과 유효한 alternative action이 필요하다. suffix의 확률성 때문에 관측한 한 쌍의 차이를 보편적인 인과효과로 부를 수 없다. v3는 variance 감소도 조건부라고 명시하며, 코드·전체 재현 artifact는 이 버전에 포함되지 않고 공개 예정이다. 논문 제목의 “free”는 replay compute 무료라는 뜻이 아니다. [§4·reproducibility checklist](https://arxiv.org/html/2609.33875v3)

**FORGE:** Lite는 추가 host probe 없이 작동하므로 모든 FORGE 비교에서 추가 LLM 호출을 가정하면 틀린다. Full의 probe 취득 비용, cached 결과와 fresh-online 결과를 분리해야 한다. Stage1→2의 개선도 action alphabet 확장 효과와 학습 효과를 나눠야 한다. 따라서 우리도 “head가 적은 토큰을 생성한다”와 “추론 전체가 빨라진다”를 별도로 검증해야 한다. [§4·부록 B/C](https://arxiv.org/html/2609.34358v1)

## 현재 pilot2의 정확한 위치 — 우리 코드에 대한 해석

현재 것은 PPO 재실험이 아니라 **공유 reference rollout으로 추정한 action return을 사용하는 근사 정책 개선**이다. 상태와 행동의 최종 가치를 다음처럼 추정한다.

\[
\widehat Q_k(s,a)=\operatorname{mean}_{j=1,2}\big[F1(\tau_{s,a,j}^{\pi_k})-\lambda C(\tau_{s,a,j}^{\pi_k})\big].
\]

여기서 continuation은 수집 시 고정한 reference \(\pi_k\)다. head learner 여섯 개는 같은 데이터를 받지만, 각 learner의 현재 정책으로 별도 수집하지 않는다. 특히 round1 reference는 round0의 MLP quality 정책이므로 상태분포가 모든 learner에 중립적이라고 주장할 수 없다. 공통 데이터 예산이라는 장점과 reference 선택 편향이라는 한계를 함께 보고해야 한다.

Return 기반 loss는 \(-\sum_a\pi_\theta(a|s)\widehat Q_k(s,a)\)에 KL·entropy 항을 더한다. 이 gradient 형태는 AggreVaTeD와 연결되지만, Q의 출처·상태 선택·작은 샘플 수가 다르므로 그 이론적 보장을 가져올 수 없다. Cost-to-go를 expert가 아닌 learner/reference suffix로 구한다는 점에서는 NRPI와 더 가깝다. [AggreVaTeD §3](https://arxiv.org/html/1703.01030v1), [NRPI §3](https://arxiv.org/html/1406.5979v1)

**Outcome CE도 미래 결과를 학습한다.** 동일한 \(\widehat Q\)에서 만든 soft target을 맞추므로 “CE=선생 모방, PG=진정한 decision”이라는 구분은 성립하지 않는다. 이번 비교가 답하는 것은 같은 미래결과 정보에서 어떤 목적함수가 품질–비용을 더 잘 일반화하는지다.

고정 Q·anchor를 가정한 이상적 목적 \(E_\pi[Q]-\beta KL(\pi\Vert\pi_0)+\alpha H(\pi)\)의 상태별 최적 분포는
\[
\pi^*(a|s)\propto\pi_0(a|s)^{\beta/(\alpha+\beta)}\exp\{Q(s,a)/(\alpha+\beta)\}.
\]
이는 해당 식을 직접 최적화해 얻는 해석이다. 우리 soft-CE의 target temperature와 추가 regularizer가 이 최적식과 동일하게 설정된 것은 아니므로 두 objective가 동일하다고 단정해서도 안 된다. 작은 parametric head, 유한 MC 표본, greedy deployment 때문에 학습 loss 감소가 전체 episode 개선을 보장하지 않는다.

## 다음 비교 실험에 반영할 항목

현재 동결 실험에 사후 끼워 넣지 않고, 다음 사전등록 계획의 우선순위로 남긴다.

1. **동일 정보·동일 예산:** teacher-action BC, outcome hard/soft CE, expected-return head를 같은 분기 데이터·동일 architecture로 비교. collection·학습·평가 비용을 모두 기록한다.
2. **분기 위치와 reference:** first-only 대 first/middle/last, 한 reference 대 learner별 reference를 별도 ablation. 단순히 뒤쪽 상태를 늘리면 좋아진다고 가정하지 않는다.
3. **Q 신뢰도:** 행동별 suffix 반복 수를 늘렸을 때 순위·return 차이가 안정되는지 train-only로 확인. 동일 상태 복원, 이미 관측된 정보만 input에 포함, forced FINAL과 자발적 종료를 구별한다.
4. **결과와 비용의 분리:** QA 유지·개선, tool calls, 실제 생성 tokens, 공급한 prefix tokens, prefill 비용을 별도 보고. 비용 감소가 오답 조기종료에서 나온 것인지 실패 사례를 조사한다.
5. **Contribution 판별:** 같은 generator의 생성형 action 선택·작은 head·간단한 종료 규칙을 비교하고, head가 미래 outcome 신호로 BC와 다른 유용한 결정을 내리는지 확인한다. all-action 분기·작은 router의 조합만으로 최초성을 주장하지 않는다.

저자 결과는 재현하지 않았으며 서로 다른 task/model/hardware의 수치를 우리 파일럿과 직접 순위 비교하지 않는다. 관련 논문의 공개 여부나 최신 버전이 바뀔 수 있으므로 ledger의 검토 날짜·버전을 기준으로 인용한다.

## 가져올 만한 방법 두 가지 — 재현이 아니라 우리 설정에 맞춘 adaptation

**A. CRR에서 가져올 것은 같은 상태의 대안과 끝까지 실행한 결과의 차이다.** 학습 질문의 prefix를 고정하고 BC가 고른 행동과 합법적인 대안 행동을 각각 실행한 뒤, 두 branch 모두 같은 frozen reference로 끝낸다. `Δ = (F1−λC)_alternative − (F1−λC)_base`를 저장한다. 먼저 이 차이가 suffix 반복에서 안정적인지 확인하고, 다음 head 학습에는 action preference 또는 action-Q로 사용한다. 비교군은 같은 횟수의 일반 rollout을 추가로 수집한 outcome-only 학습과 같은 branch 데이터의 soft CE다. 우리 기본형은 CRR의 token-level advantage replacement·LoRA·SWE 환경을 재현한 것이 아니라, categorical action head로 옮긴 adaptation이다. 외부 상태를 바꾸는 도구로 확장할 때는 환경 snapshot 복원부터 검증해야 한다. [CRR §4–5](https://arxiv.org/html/2609.33875v3)

**B. FORGE에서 가져올 것은 outcome을 버리지 않는 soft-target warm start다.** 학습 상태별 모든 합법 행동의 실제 `Q=정확도−λ비용`을 수집하고, 명시한 temperature로 `target(a)=softmax(Q/τ)`를 만든다. 같은 frozen hidden feature·같은 head 용량에서 hard-best-action CE, soft CE, expected-return 학습을 비교한다. 이것이 teacher가 고른 행동 한 개를 모방하는 BC보다 적절한 강한 기준선이다. τ·λ는 train/validation에서 사전 고정하며 평가 정답·평가 rollout으로 target을 만들지 않는다. 현재 pilot2가 이미 이 방향 일부를 구현한다. FORGE의 증거형식/thinking action·BGE feature·GRPO를 그대로 재현한 것으로 부르지 않는다. [FORGE §2.2–3.2](https://arxiv.org/html/2609.34358v1)
