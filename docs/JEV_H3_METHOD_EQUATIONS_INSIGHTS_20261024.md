# H3: 학습 방법·수식·결과·인사이트

2026-10-24. 실제 실행된 H3를 기록한다. 이후 후보나 제안은 H3의 구현 사실과 구분한다.

## 1. 질문과 개입

H1은 결과를 배운 MLP 행동 정책이다. H2는 품질이 조금이라도 좋아지면 계속하는 종료 선호를 학습해 F1은 올렸지만 비용이 늘었다. H3의 질문은 **학습 자료의 품질 하한을 유지하면서 남은 생성 비용을 최소화하는 종료 판단을 배울 수 있는가?**이다.

Qwen3-4B-Instruct-2507, native A-RAG 프롬프트·도구·인자·관측·디코딩·한도는 고정했다. 행동은 keyword_search, semantic_search, read_chunk, FINAL이다. LLM과 H1 MLP 전체를 동결하고 FINAL logit의 상태별 보정 129개 파라미터만 학습했다. 도구 사이의 순위는 같은 입력 상태에서 그대로다. 종료 결정이 달라지면 이후 방문 상태와 전체 경로는 달라질 수 있다.

## 2. 데이터와 반사실 비교

고정 train192 중 48문항에서 H1 greedy 정책이 실제 방문한 상태를 사용했다. 각 상태 s에서 두 가지 개입을 비교했다.

- a=0: H1이 가장 선호하는 합법적인 비종료 도구를 실행하고, 이후 H1 greedy로 진행한다.
- a=1: 지금 FINAL을 실행한다.

두 개입 모두 실제 답변을 생성하여 최종 answer F1 Q(s,a)와 남은 생성 토큰 C(s,a)를 기록했다. 미래 정답/F1은 학습 타깃에만 사용한다. 실행 시 정책 입력에는 미래 결과나 gold가 없다. 동일 greedy 개입은 한 번씩 실행했다. 따라서 확률적 반복성까지 입증한 자료는 아니다.

처음 83개 시작/마지막 상태에서는 수정 목표가 1문항뿐이라 최소 3문항 기준에 미달했다. 저장된 경로의 중간 상태를 보완해 **48문항, 137상태 = 83 + 54**가 됐다. 도중 기존 snapshot 함수가 중간 상태 하나만 반환해 34상태만 수집한 실패가 있었다. 독립 검증에서 발견해 결과를 보존하고 빠진20상태만 추가했다. 중간 수집 총108 suffix episodes, 5200생성/319427입력 토큰, 오류0. 추가 수집과 타깃 변경을 함께 적용했으므로 각 기여를 독립적으로 분리한 실험은 아니다.

137개 중간/초기/마지막 상태의 suffix는 서로 겹친다. 이들의 가중 비용은 **학습용 대리 지표**이며, 48개 전체 episode의 실제 총비용이나 정확도가 아니다.

## 3. 품질 하한을 둔 비용 최소화 타깃

문항 q의 상태 수를 n_q, 문항 수를 N이라 두고 상태별 가중치를 다음과 같이 정했다.

$$w_s = \frac{1}{N n_q}.$$

문항마다 같은 총 가중치를 주고, 문항 안에서는 상태에 균등하게 배분한다. H1의 기존 종료 선택은 x_s^0∈{0,1}, 새 학습 타깃은 x_s∈{0,1}이다.

$$Q_s(x_s)=(1-x_s)Q(s,0)+x_sQ(s,1),$$
$$C_s(x_s)=(1-x_s)C(s,0)+x_sC(s,1).$$

타깃은 다음 이진 최적화로 정했다.

$$\min_{x} \sum_s w_s C_s(x_s)$$
$$\text{subject to}\quad \sum_s w_s Q_s(x_s) \ge \sum_s w_s Q_s(x_s^0).$$

추가 제약은 다음과 같다.

- 어느 개입이든 실행 오류가 있으면 기존 선택을 유지한다.
- 두 개입 모두 F1=0이면 기존 선택을 유지한다. 싼 오답을 긍정적인 학습 타깃으로 삼지 않는다.
- 품질과 비용이 모두 같으면 기존 선택을 유지한다.
- 기존 선택의 F1>0인 상태에서 F1=0인 선택으로 바꾸지 않는다.
- 양의 F1끼리의 손실은 다른 상태의 이득으로 상쇄할 수 있다. **개별 상태의 품질 보존 제약은 아니다.**

동률은 비용 최소화 → 품질 최대화 → 가중 변경 수 최소화 순서로 처리했다. SciPy MILP를 단계별로 풀었으며 수치 허용오차는 구현에 명시돼 있다. 타깃 자체는 미래 결과를 아는 학습용 최적 선택이지 배포 가능한 oracle 정책 성능이 아니다.

결과: 15상태/11문항에서 기존 선택과 다른 타깃을 얻었다. 품질 개선3, 품질 동일10, 양의 품질 하락2상태다. 종료 타깃은10문항, 계속 타깃은1문항에 걸쳐 있다. 가중 suffix F1은0.450562→0.457159, 생성은52.9608→49.6684였다.

## 4. 상태별 종료 보정 head

H1의 입력은 저장된 hidden pack의 last-token, question-end, observation-mean 세 벡터를 연결한 값이다. 고정된 평균/스케일로 정규화하고 기존 MLP의128차원 은닉 표현을 얻는다.

$$z_s=\tanh\left(W_1\frac{h_s-\mu}{\sigma}+b_1\right),\qquad \ell^{H1}(s)=W_2z_s+b_2.$$

H3는 다음과 같이 FINAL logit만 보정한다.

$$\ell^{H3}_a(s)=\ell^{H1}_a(s),\quad a\in\{keyword,semantic,read\},$$
$$\ell^{H3}_{FINAL}(s)=\ell^{H1}_{FINAL}(s)+u^\top z_s+b.$$

학습 파라미터는 u∈R^128와 b∈R, 합계129개다. 0으로 초기화하므로 시작 정책은 H1과 정확히 같다. W1,W2,b1,b2,μ,σ와 LLM은 동결했다. 실행은 합법 행동 중 logit argmax이며, 별도 종료 확률 임계값을 개발 데이터에 맞추지 않았다.

이 식은 상태 의존적인 **종료 선호**를 표현한다. uᵀz+b를 보정된 정답 확률, 실제 기대 정보 가치 또는 비용 예측이라고 해석하면 안 된다.

## 5. 실제 학습 손실

합법 도구 중 가장 큰 logit과 FINAL의 차이를 종료 margin으로 정의한다.

$$m_\theta(s)=\ell^\theta_{FINAL}(s)-\max_{a\ne FINAL,\ a\ legal}\ell^\theta_a(s).$$

여기서 p_θ(s)=sigmoid(m_θ(s))는 binary surrogate다. 4개 행동 softmax의 FINAL 확률과 같지 않다. 수정 타깃이 있는 상태 집합 D_C에서는 기존 H1 margin 확률과 이진 최적 타깃을 절반씩 섞는다.

$$t_s=\tfrac12\operatorname{sigmoid}(m_{H1}(s))+\tfrac12x_s^*.$$

나머지 새 상태와 과거 학습 특징 중 중복을 제거한 유지 집합 D_R에서는 H1의 기존 margin 확률을 유지 타깃으로 사용한다. 과거 서로 다른 정책의 outcome을 새 H1의 값으로 섞지 않고, 과거 자료는 상태 특징 보존 용도로만 쓴다. 실제 수정15상태, 유지510상태였다.

$$L(\theta)=\tfrac12\sum_{s\in D_C}\tilde w_s\,BCEWithLogits(m_\theta(s),t_s)
+\tfrac12\sum_{s\in D_R}\bar w_s\,BCEWithLogits(m_\theta(s),\operatorname{sigmoid}(m_{H1}(s))).$$

각 집합 안에서 문항별 동일 총 가중치가 되도록 정규화했다. 두 집합은 상태 수와 무관하게 손실의 절반씩을 차지한다.

| 설정 | 실행값 |
|---|---|
| optimizer | AdamW, PyTorch 기본 weight_decay=0.01 |
| learning rate | 0.001 |
| 전체 batch 업데이트 | 100회 |
| seed | 20261069 |
| gradient norm clipping | 1.0 |
| 학습 파라미터 | FINAL 보정129개 |
| epoch/계수 탐색 | 없음 |

동결 파라미터 동일성, 같은 입력에서 도구 logit 동일성, 유한값, checkpoint 재로드 동일성을 검증했다. correction/retention loss 감소는 실행 증거이며 QA 성능 입증이 아니다.

## 6. 실제 QA 결과

같은 개발64문항에서 끝까지 실행했다. 적응적으로 여러 후보를 본 개발 자료이며 독립 검증이 아니다.

| 정책 | F1 (%) | 평균 생성 토큰 |
|---|---:|---:|
| 고정 기본 GRPO |22.50|98.41|
| H1 |28.30|75.34|
| H2 |29.61|79.28|
| H3 |27.32|66.89|

H3는 H1 대비 F1−0.9884%p/생성−11.22%, 고정 기본 대비 F1+4.8152%p/생성−32.03%다. 목표 F1≥27.5007/생성≤68.8844 중 비용만 통과했다. H1 대비 F1 개선3문항, 악화4문항이다. F1 차이의 paired marginal95% CI는 약[−6.77,+5.31]%p로 넓다. 차이를 확정적인 일반화 성능으로 주장하지 않는다.

| H1→H3 결과 유형 | 문항 수 | 순절약 생성 토큰 |
|---|---:|---:|
| F1 개선 |3|33|
| 양의 F1 유지 |17|60|
| F1 하락 |4|143|
| 둘 다 F1=0 |40|305|
| 합계 |64|541|

양의 F1은 완전정답과 같지 않다. 위 표는 그룹별 순합이며 모든 해당 문항이 절약했다는 뜻도 아니다. 유용한 절약93토큰이 존재하지만, 총절약의 대부분은 둘 다 오답이거나 품질이 악화된 문항에서 나왔다.

128개 평가 episode/183.92초, 생성9103/입력453629, 오류0. H1reference64개는 이전 결과와 답변/F1/토큰이 정확히 같았다. 원본 결과와 Drive archive bytes를 검증했다. 공유 GPU에서 통제된 latency 우월성을 주장하지 않는다.

## 7. 인사이트와 아직 모르는 것

**확인된 사실**

1. 시작·종료만으로 놓쳤던 중간 종료의 학습 근거가 있었다. 범위를 채우니 변경 근거가1문항에서11문항으로 늘었다.
2. 작은 FINAL 보정만으로 실제 행동 경로와 품질/비용 균형이 달라졌다. 큰 모델 변경 없이 계산 배분을 조절할 수 있다.
3. H2의 국소 품질 우선 목표와 달리 H3는 비용을 줄이는 후보를 만들었다. H3는 버릴 실패가 아니라 보존할 비용 중심 기준 후보다.
4. 학습 상태들의 평균 품질 하한은 새 문항의 품질 보존을 보장하지 않는다. H3는 학습에서 일부 양의 품질 손실을 허용했고 평가에서도 손실이 있었다. 인과 원인이 그것 하나라고 증명한 것은 아니다.

**아직 미입증인 주장**

- H3의 이득이 상태별 보정 덕분인지, 단순히 더 자주 종료하는 성향 덕분인지.
- 129개 파라미터가 일반적인 정보 가치나 정답 충분성을 학습했는지.
- 다른 모델/데이터/검색 환경으로 전이되는지, 새 독립 문항에서도 공동 목표를 달성하는지.
- 중간 자료 확대와 타깃 변경 각각의 독립 기여.

최신 사용자 승인에 따라 **H3를 고정하고, 상수 종료 보정 대조군과 비교**한다. H4 품질 개선 확장보다 이 검증을 먼저 한다. H1 기반·도구 순위를 유지하며 uᵀz+b 대신 학습 자료에서 비용만으로 보정한 상수를 적용한다. 이 대조군도 H1 자체의 상태 의존성을 유지한다. 따라서 정확한 질문은 “H3가 추가한 상태 의존적 종료 보정이 상수 보정보다 유용한가?”이며 “전체 정책이 상태를 보느냐?”를 완전히 제거하는 실험은 아니다.

상수는 학습 문항에서 H3와 유사한 실제 토큰 비용을 목표로 정하고 고정한다. 개발 결과를 보고 재조정하지 않는다. 비용이 충분히 비슷하지 않으면 단정하지 않는다. 유망하면 이후 학습 범위/독립 확인을 확장한다. 큰 ablation 행렬이나 개발 penalty sweep은 하지 않는다.

## 8. 재현·복구 위치

- 타깃 코드: `src/lsdrag/decision_budget_targets.py`
- 보정 구조/손실 margin: `src/lsdrag/decision_stop_value.py`
- 학습 실행: `scripts/fit_decision_budget_stop_h3.py`
- 고정 계획과 평가 spec: `lab/registry/decision_budget_stop_h3_20261024/`
- 원래 프로토콜: `docs/JEV_BUDGET_STOP_H3_20261024.md`
- 결과/원본 검증: `lab/reports/decision_budget_stop_h3_validation64_20261024/`
- 중간 수집 실패 및 보완 검증: `lab/reports/decision_stop_middle_h3_20261024/`
- 결과 요약: `docs/JEV_BUDGET_STOP_H3_RESULT_20261024.md`
- H1 checkpoint SHA256: `32849a288d13e53c32bf6b0e950d8dea791e36e0d0aa41114376bcbe01f9658a`
- H3 checkpoint: `/content/local_fast/lsdrag-text-v1/results/decision_budget_stop_h3_20261024/stop.pt`
- H3 SHA256: `936e45925890f1a1da6a411719a092db7fc68f60aee6fd2b68e15e65cb0daf94`
- 결합 학습 자료: `/content/local_fast/lsdrag-text-v1/results/decision_budget_stop_h3_inputs_20261024/`; provenance에3개 source tensor 해시가 있다. 부모 archive로 재구성할 수 있다.
- Drive checkpoint archive: `/content/drive/MyDrive/lsdrag-backups/20261023/decision_budget_stop_h3_20261024.tar.gz`; 해시는 해당 registry의 `archive.json`.
- Drive 평가 archive: `/content/drive/MyDrive/lsdrag-backups/20261023/decision_budget_stop_h3_validation64_20261024.tar.gz`; 해시는 해당 registry의 `archive.json`.
- 기존 결과까지 코드 백업: `/content/drive/MyDrive/lsdrag-backups/20261024/code_h3_result_19d60138.tar.gz`, SHA256 `7f342dc8557fc3941c09a1cf5614f65b7a5ce2a9b733d7128429d03eaf0aeba9`.

기존 run 이름은 불변이다. 재실행하려면 새 namespace와 동일 입력 해시를 명시해야 한다. 기존 스크립트의 디렉터리 존재 방지 장치를 제거해 덮어쓰지 않는다. 코드/메모/프로토콜은 jev_v1.1에 push하고 모델·대형 자료는 Git에 넣지 않는다. Goal ACTIVE, reserved80 미사용.
