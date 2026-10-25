# Microsoft Decision-1 원리 벤치마킹 — API 사용 없음

사용자 의미는 API 성능 실측이 아니라 원리 조사와 우리 연구 설계에 대한 벤치마킹이다. API 연동으로 해석한 것은 잘못이었다. 추론 호출0회, 준비된 호출 스크립트는 삭제했고 자격증명은 필요 없다. 공개 문서/모델 목록을 조회한 것은 추론 실험이 아니다.

## 공개 사실과 확인되지 않은 내용

| 항목 | 확인 내용 | 우리 연구에 대한 함의 |
|---|---|---|
| 기반 모델 | Qwen3.5-9B를 decision scoring용으로 후학습 | 같은 backbone 계열을 쓰지만 학습 목표·가중치가 같다는 뜻은 아님 |
| 출력 | 정해진 선택지를 single-pass로 점수화 | 행동 이름을 토큰으로 길게 생성할 필요 없음 |
| 목표 | 선택 정확도·확률 보정·입력/선택지 표현 변화에 대한 견고성 강조 | 빠른 dispatch와 판단 품질을 별도로 검증해야 함 |
| 불명확 | head 상세구조, 학습데이터 구성, 정확한 loss, 동결/LoRA/full FT 범위, 다중턴 RL 사용 여부 | head-only 학습의 충분성이나 장기정책 학습을 MS사례로 입증할 수 없음 |

위 공개 설명의 근거: [Microsoft 공식 발표](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/)와 [Foundry Labs 소개](https://labs.ai.azure.com/innovations/microsoft-decision-1/). 문서에 없는 구조를 임의로 특정하지 않는다. Foundry Labs의 “CODE GitHub Repository” 링크도 확인 시 실제로는 발표 블로그를 가리켰으며 모델 학습 소스 공개의 증거가 아니다.

공식 문서가 설명하는 타입은 `choice`(상호배타적 선택), `noul`(참/거짓 확률), `score`(순서척도)다. 신뢰도는 과제에 따라 달라질 수 있고, 상태·선택지 문구/순서 영향도 평가하도록 안내한다. [공식 사용 문서](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision)

## 우리의 핵심 구분

현재 action head는 다음 분포를 낸다.

\[\pi_\theta(a\mid s)=\operatorname{softmax}(z_\theta(s))_a.\]

이는 정책의 행동 선택 확률이다. `read_chunk` 확률0.9는 그 행동 후 정답확률90%라는 뜻이 아니다. 이 분포의 temperature를 조정해도 argmax는 그대로라서, greedy정책의 정확도·토큰효율이 자동으로 좋아지지 않는다.

우리가 필요한 값은 예를 들어 다음이다.

\[q_\pi(s,a)=P(\mathrm{EM}=1\mid s,\operatorname{do}(a),\text{이후 정책 }\pi),\qquad c_\pi(s,a)=E[T_{\mathrm{remaining}}\mid s,\operatorname{do}(a),\pi].\]

여러 행동이 모두 정답에 도달하거나 모두 실패할 수 있으므로 q는4행동의합이1일 필요가 없다. 각각의이진결과 예측으로 학습할 수 있다. 이는 **MS내부구현의주장아닌 우리의추가설계**다. 입력에 미래결과나gold를 주는 것이 아니라 학습label에만 최종EM/비용을 쓴다.

기존 value_aux는 F1/비용 예측을 표현학습의 보조loss로 사용했고 실제행동은 별도policylogits로 골랐다. 따라서 그 결과를 “예측된 최종성공/비용을 직접 고려해 선택했다”고 해석하면 안 된다. 예측값으로 행동을 바꾸는 설계는 별도 검증이 필요하다.

## 기존 결과와 연결

- 공유9B에 작은head로 선택과인자생성을 연결하는 구현은 이미 있다. MS의 single-pass개념만 따라한다고 핵심문제가 풀리지는 않는다.
- 종료H3는 단순bias보다 우위가명확하지 않았고 공통정답비용은 증가했다.
- value_aux는 기존정답24개를유지하고1개를더맞혔지만 전체토큰+0.80%,공통정답+0.23%였다. 추가정보수집의약한신호이지효율성공은아니다.
- GRPO스모크는같은오답을짧게내는업데이트를보였다. 모든오답인그룹에서도 작은길이차이가표준화후약±1이된다는목표문제가있다.

따라서 벤치마킹할 원리는 **상태에 조건부인 명확한 판단 문제 정의, 결과 의미가 있는 점수, 그 점수의 검증**이다. MS가장기정책을이미입증했다고가정하거나API모델을추가해두모델구성으로전환하지않는다. 추가실험은 `JEV_DECISION_FOLLOWUP_DESIGN_20261025.md` 참조.
