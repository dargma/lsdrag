# Decision module과 RAG: 품질·전체 비용 관점 재평가

2026-10-15. 사용자 요청: 직전 발언에 맞춰 결론을 바꾼 것은 아닌지 점검하고, 인자 최소화/인터페이스 재설계를 포함해 RAG와 decision module의 결합을 평가한다. 문헌과 코드의 정적 검토이며 새로운 실험 결과가 아니다. 확정된 전체 목표(학습·dev 평가 포함)는 유지하되 아래 설계 판단은 새 실행 계약으로 아직 동결하지 않았다.

## 판단의 수정

앞선 답변은 인자 최소화 제안에서 바로 실행 계획으로 넘어갔고, 복잡성 우려를 받은 뒤에는 인터페이스 재설계 쪽을 너무 빨리 추천했다. 어느 쪽도 현재 비용/실패 원인 계측으로 선택된 결론이 아니다. 현재 타당한 결론은 decision module을 실험할 근거는 있지만, 전면 재설계나 인자 제어기 추가가 필요하다는 증거는 없다는 것이다. 효율은 도구 호출/생성 토큰 하나가 아니라 답변 품질과 전체 계산량의 관계로 평가한다.

## 기존 연구가 뒷받침하는 범위

| 연구 | 확인한 기능과 시사점 | 우리 주장에 대한 한계 |
|---|---|---|
| [Adaptive-RAG, NAACL2024](https://aclanthology.org/2024.naacl-long.389/) | 질문 복잡도에 맞춰 서로 다른 retrieval 전략을 선택 | 작은 routing classifier의 가능성이지 가변 다중모달 행동/인자의 판단 가능성 증명은 아님 |
| [Self-RAG, ICLR2024](https://arxiv.org/abs/2310.11511) | 학습된 reflection tokens로 retrieval/생성/평가를 결합 | frozen backbone의 작은 추가 head만으로 같은 능력이 나온다는 결과가 아님 |
| [DRAGIN, ACL2024](https://aclanthology.org/2024.acl-long.702/) | 정보 필요에 맞춰 retrieval 시점과 query를 결정 | when과 what을 함께 봐야 하며, query 생성을 삭제하면 된다는 근거가 아님 |
| [SetR, ACL2025](https://aclanthology.org/2025.acl-long.861/) | 개별 relevance 외에 정보 요구를 공동 충족하는 passage set 선택 | 후보 ranking과 증거 집합의 충분성/다음 행동 효용은 다름 |
| [Search-R1,2025](https://arxiv.org/abs/2503.09516) | 실제 검색 rollout과 outcome reward로 reason/search 정책 학습 | 생성 정책의 RL이며 단순 head의 feasibility나 총 비용 감소를 대신 증명하지 않음 |
| [Sufficient Context](https://arxiv.org/abs/2411.06037) | context 부족과 충분한 context 활용 실패를 구분 | 충분한 evidence와 현재 reader의 정답 가능성은 동일하지 않음; selective accuracy는 응답 coverage와 함께 평가해야 함 |
| [HALT,2026 preprint](https://arxiv.org/abs/2608.02009) | evidence coverage 기반 stopping; generated claims와 gold diagnostic을 구분 | stopping 자체가 새로운 기여는 아니며 query-decomposition 및 verification 비용이 남음 |
| [When Should Multi-Round RAG Stop?,2026 preprint](https://arxiv.org/html/2608.13237v1) | first STOP에 의해 결정되는 순차 정책을 평가. 보고된 검색 호출 감소3.70%, EM 변화−0.625%p | 저자가 총 inference 비용 감소를 입증하지 않았다고 명시. judge 비용과 sequential false-stop 위험을 무시하면 안 됨 |

초록/공식 논문 페이지와 A-RAG 및 stopping 논문 관련 본문을 확인한 범위다. 전체 재현/신규성 전수조사는 아니다. 여러 논문의 수치가 서로 다른 backbone/data/metric이므로 순위를 매기지 않는다. 2026 preprint의 심사 완료를 가정하지 않는다.

## A-RAG를 유지해야 할 이유와 바꿀 수 있는 경계

[A-RAG §3–5](https://arxiv.org/html/2602.03442v1)는 keyword/semantic/chunk read를 통해 점진적으로 증거를 확보한다. 논문은 복잡한 orchestration을 피한 단순 loop를 사용하며 모델의 reasoning과 탐색 예산도 성능에 영향을 준다고 보고한다. 따라서 읽기와 검색을 자동 병합하거나 새 query를 제한하는 것은 단순 serialization 최적화가 아니라 정책 능력의 변화다. 우리의 멀티모달 추가 도구와 G1 실제 이미지 전달도 유지해야 한다.

안전한 최소 경계는 기존 backend/행동 의미를 유지하고 선택 표현만 바꾸는 것이다. ID string 대신 runtime slot으로 동일 ID를 전달하는 것은 가능하지만, 제한된 shortlist, 단일 ID 강제, 전체 corpus 후보 열람, 새 query 제거는 별개의 변화다. Joint(action,target) 선택은 추가 도구선택+인자선택의2회 판단을 줄일 가설이지만, 가변 후보 scoring·학습 label·공통 후보 baseline이 필요하다. 현재6-class linear checkpoint를 그대로 쓸 수 없다. 지금 필요성이 입증된 상태가 아니다.

## 삽입 위치별 품질/효율의 기대와 실패

| 위치 | 품질이 좋아질 경로 | 비용이 줄어들 경로 | 주된 위험 | 현 단계 판단 |
|---|---|---|---|---|
| 질문별 retrieval 전략 routing | 쉬운/어려운 질문에 맞는 경로 | 쉬운 질문의 반복 검색 회피 | 최초 질문만으로 후속 정보 필요를 알기 어려움 | 유용한 단순 비교군; 멀티모달 순차제어의 대체는 아님 |
| 기존 여섯 행동 선택 | 잘못된 도구/조기FINAL 교정 | 생성 결정/불필요한 회차 감소 가능 | backbone 비용과 query 비용 존속; 모방은 최적화가 아님 | 현재 코드로 가장 분리해서 검증 가능 |
| 증거/이미지 후보 선택 | distractor 대신 필요한 증거 채택 | 긴 context·불필요한 vision 입력 감소 가능 | 후보 밖 정답 경로 누락; image embedding만으로 detail 판단 어려움 | 의미 있으나 후보 제한의 효과를 분리해야 함 |
| evidence-aware stopping | 충분한 근거에서 답하거나 부족할 때 더 탐색 | 불필요한 후속 검색 방지 | premature stop; judge 비용; 자신감은 충분성 아님 | 개선 여지를 먼저 계측; 최우선으로 단정하지 않음 |
| 인자 후보 재사용 | ID/형식 오류 감소 | ID/JSON decode 생략 | 선택 prefill 추가, query 재사용 실패, 재사용 자체가 goal로 변질 | 보조 ablation 후보, 주된 품질 개선 가설은 아님 |
| 전체 planner 재설계 | 선택지와 정보 흐름을 제어에 맞게 정리 | controller 호출 축소 가능 | 기능 제한/새 학습 부담/효과 귀속 어려움 | 현재는 보류; 유지형이 실패한 이유가 인터페이스임을 확인한 뒤 |

## 작은 head에 대한 정확한 가설

현재 `head_experiment.py`는 full multimodal backbone의 마지막 norm/마지막 token 표현에서 linear6를 학습한다. 큰 모델의 상태에 적절한 정보가 있으면 작은 readout이 이용할 수 있다는 가설이다. 작은 독립 모델이 새로운 reasoning을 해결한다는 주장은 아니다. `direct_reader.py`는 별도 decision prefill 뒤 argument/answer prefill을 수행한다. head가 작다고 전체 모델 호출이 사라지지 않는다.

현재 목적함수는 teacher action CE다. 원래 teacher의 잘못된 판단과 불필요한 호출도 모방한다. 개선이 우연히 발생할 수 있지만 품질/비용 개선을 직접 학습하는 target이 아니다. trained head는 같은 state의 first-token action readout보다 나은지도 확인해야 한다. 약한 generated-letter teacher만 비교하면 통제 형식의 약점을 이긴 결과가 될 수 있다. Native tool baseline은 동일 모델/backend/도구/예산/출력조건에서 따로 검증한다.

우리가 지향할 최종 판정은 고정 비용 예산에서 더 높은 QA 품질, 또는 고정 품질 기준에서 더 낮은 총비용이다. Calls/argument tokens는 기전 지표다. 현 shared GPU에서는 관측 비용 항목과 token/호출 수를 공개하되 latency 우월성 결론은 보류한다. 최종 answer tokens, candidate 생성/선택, fallback, 추가 prefill/vision, retrieval, training 비용을 숨기지 않는다.

## 현재 증거가 말하는 것

`lab/reports/full_head_20261015_v2/full_head_progress.json`의 과거20문항 checkpoint는 valid answer19/20, EM0.05, F10.1105, image-delivered questions17/20을 기록한다. 이는 중단된 generated-letter/commonargs 파일럿의 보고서이며 이번에 재실행한 수치가 아니다. Original native A-RAG의 성능 또는 head가 실패한 결과로 해석할 수 없다. Label/result를 새 훈련에 재사용하지 않는다. 이 수치는 기능/형식 성공이 품질을 보장하지 않는다는 점과 baseline diagnosis 필요성을 보여준다. 이미지가 전달됐다는 사실도 답변에 필요한 이미지가 정확히 읽혔다는 증거가 아니다.

## 다음 실행에서 분리해야 할 검증

1. **Baseline 신뢰성:** 같은 현재 도구환경에서 native 또는 검증된 원형 생성 정책과 controlled first-token의 품질을 확인. 검색 결과 누락, 잘못된 entity/query, 잘못된 읽기/이미지, 성급한FINAL, evidence가 있어도 답변 실패, 형식/인프라 오류를 구분한다. 근거만 제공하는 별도 reader 진단은 oracle 진단으로 표시하고 정책 입력에 gold를 주지 않는다.
2. **행동의 개선 여지:** outcome-blind로 정한 train states에서 동일 reader/continuation/budget 하에 기존 행동과 대안 행동을 비교한다. 즉시 F1만 보면 multi-hop의 bridge action을 잘못 낮게 평가하므로 제한된 동일 후속 정책의 결과까지 포함한다. 학습용 정답은 rollout 뒤 scoring에만 쓴다. 실제 다른 행동이 도움이 되는지와 작은 readout이 그 차이를 구분하는지는 별도 질문이다. Oracle은 제한된 비교 대상 안의 진단이지 전역 최적 정책이 아니다.
3. **작은 head의 추가 가치:** 같은 interface/args 조건의 first-token과 learned head를 비교. BC는 읽기 가능성/모방 기준선; 성능 개선을 주장하려면 outcome/cost에 연결한 별도 target 또는 개선 사례의 증거가 필요하다. Dev split은 selection/tuning에 재사용하지 않는다.
4. **인자 최소화의 순증분:** 앞선 controller를 고정하고 동일 관측 ID의 생성/slot 선택만 비교하는 ablation부터. 새 query/문서범위/다중읽기 기능은 유지한다. 추가 prompt/prefill까지 세고, 후보를 제한해야 한다면 같은 후보의 생성형 기준선을 둔다. Query reuse controller와 joint pointer와 cache를 동시에 추가하지 않는다.

실제 branch 데이터 수집/새 target 학습은 기존 BC 계획과 다른 protocol이므로 manifest/commands/gate를 수정해 동결한 뒤 실행한다. 이 리뷰가 묵시적으로 이미 승인된 run identity를 변경하는 것은 아니다. 단순한 현재 readout feasibility와 새로운 연구기여를 구분한다.

## 권고

Decision+RAG 연구는 타당한 후보이지만 decision 추가 자체, tool call 감소 자체, 인자 토큰 감소 자체는 품질/전체 효율 개선의 증거가 아니다. 지금 선택은 **기존 A-RAG 환경에서 decision의 추가 가치부터 검증하고, 인자 최소화는 독립 ablation으로 유지**하는 것이다. 첫 단계부터 후보선택기·query재사용기·stopping verifier를 함께 붙이지 않는다. 병목 분석이 evidence selection을 가리키면 그 부분을, stopping을 가리키면 그 부분을 연구하며 현재 사용자 발언만으로 어느 쪽도 우선 확정하지 않는다. 효과가 없다면 decision module이 불필요하다는 결과도 받아들인다.
