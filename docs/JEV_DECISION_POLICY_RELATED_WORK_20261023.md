# Decision policy head: 최신 related work와 벤치마킹 방향

조사 기준일: **2026-10-23**. 사용자 요청에 따라 head/표현, trajectory 학습, 효율적 RAG를 세 갈래로 병렬 조사하고 원 논문·저자 공개 자료를 확인했다. 최신 검색으로 확인한 선택적 조사이며 전수조사나 논문 재현 완료 보고가 아니다. 원문 검증 깊이·버전·미확인 사항은 연결된 세부 노트와 JSON source ledger에 남긴다. 2026-10-17 조사 문서는 역사 기록으로 보존한다.

## 결론과 제안하는 contribution

**작은 head, frozen LLM, 확률 출력, trajectory 결과 기반 학습 각각은 이미 선행이 있다.** 우리 연구의 검증 대상은 다음의 구체적 결합이다.

> 별도 decision backbone 없이, 생성 LLM의 현재 prefill 표현으로 매 단계 도구/종료를 선택하고 같은 cache에서 인자·답변 생성을 이어간다. 작은 head를 행동별 후속 품질·비용으로 학습하여 강한 native/모방/단순 종료 기준보다 나은 실제 quality–cost를 얻는지 검증한다.

비슷한 연구가 실제로 작동했다는 것은 구성 요소의 구현 가능성을 뒷받침하며, 검증된 방법과 실패 조건을 배울 근거다. 다만 작은 데이터·frozen4B·head-only라는 우리의 조합에서 성공할 확률을 수치로 추정할 근거는 없다.

이는 **차별화 가설**이지 최초성·우월성 선언이 아니다. 같은 prefill reuse의 시스템 효과와 행동정책 학습의 효과를 따로 입증해야 한다. 논문마다 backbone·데이터·도구·예산이 달라 보고된 최고점끼리 직접 순위를 매기지 않는다.

## 가장 가까운 선행 지도

| 연구·확인 버전 | 이미 있는 요소 | 우리에게 필요한 비교·학습점 |
|---|---|---|
| [LAnR v3](https://arxiv.org/html/2604.17866v3), 2026-10-19 개정 | 같은 hidden에서 작은 retrieve/stop control | same-forward head 자체를 novelty로 삼지 않는다. 고정 backbone과 terminal outcome 학습의 이득을 분리한다. |
| [FunDM](https://arxiv.org/abs/2610.17175), 2026-10-12 | frozen 표현의 작은 head로 teacher 확률 증류 | 증류 head를 강한 기준으로 유지한다. 더 복잡한 구조보다 실제 결과 학습이 도움이 되는지 비교한다. |
| [OLIVIA](https://arxiv.org/html/2605.11169v1), 2026-05-11 | frozen hidden 기반 action bandit·불확실성 탐색 | 작은 선형 정책도 비교 대상이다. 즉시 도구 라벨과 후속 QA return은 서로 다른 신호다. |
| [FORGE](https://arxiv.org/html/2609.34358v1), 2026-09-28 | frozen host+작은 router, outcome/cost target | 같은 rollout으로 soft CE와 return 목적을 비교한다. 외부 feature/probe 비용도 센다. |
| [CIPO](https://arxiv.org/html/2610.04991v1), 2026-10-04 | 같은 상태에서 대안 행동 후 continuation 결과 비교 | action library 변경과 학습 알고리즘 개선을 구분한다. 현재 네 행동은 고정한다. |
| [CRR v3](https://arxiv.org/html/2609.33875v3), 2026-10-03 개정 | 실행 가능한 환경 분기에서 step-level return contrast | replay 비용 포함 동일예산 비교와 state 복원 정확도를 배운다. |
| [Stop-RAG](https://arxiv.org/html/2510.14337v1), 2025-10-16 | STOP/CONTINUE의 미래 품질·비용 | 여러 도구 정책의 이득인지 단순 종료 개선인지 분리한다. |
| [GRASP](https://arxiv.org/html/2607.10463v1), 2026-07-11 | keyword/semantic/read/answer 및 효율 보상 | 행동 공간 자체도 선행이다. backbone 전체 학습 대비 작은 head의 학습·추론 비용을 검증한다. |

날짜는 최초 공개일과 개정일을 구분했다. 특히 LAnR은 기존 저장 v2 이후 **10월19일 v3**가 확인됐다. D3/Decision-1 계열의 공개 런타임·학습 데이터 범위는 [head 노트](research/20261023/head_related.md), 추가 검색/종료 연구는 [RAG 노트](research/20261023/efficient_rag_related.md)에 정리한다.

## 좋은 방법을 가져오는 순서

아래는 논문 요소를 **우리 환경에 맞게 적용하는 비교 설계**다. 동일 논문 데이터·모델·예산을 그대로 재현하는 실험과 구분한다. 현재 pilot2에 사후로 새 조건을 끼워 넣지 않는다.

| 우선순위 | 가져올 방법 | 우리 실험에서 고정할 것 | 판별할 질문 |
|---|---|---|---|
| 1 | FORGE의 utility 기반 soft target, CRR의 상태별 return 비교 | 동일 state/action/suffix 표본, 같은 reference, head 초기값·업데이트 수 | 모방보다 낫나? 같은 정보로 학습한 CE보다 return 목적이 낫나? |
| 2 | Stop-RAG에서 배운 종료 전용 대조군 | BC의 non-FINAL 도구 선택은 고정, binary head만 STOP/CONT 선택 | 비용 감소가 전부 일찍 종료한 효과인가? 네 행동 학습이 추가로 필요한가? |
| 3 | OLIVIA에서 배운 선형 uncertainty-aware 선택 | 같은 hidden·네 행동, train에서만 feedback/update, 같은 수집 예산 | 작은 선형 모델과 탐색 보강만으로 충분한가? |
| 4 | FunDM/typed decision의 선택지 표현 readout | 먼저 last-linear/MLP와 같은 데이터·parameter/연산 예산 | head 구조가 병목인가, 학습 target/coverage가 병목인가? |

1번은 **현재 pilot2가 일부 수행 중**이다. 2–4번은 후속 구현·평가 대상이며 완료하지 않았다. OLIVIA의 oracle tool-set reward를 그대로 가져오지 않고 우리가 실제로 관측 가능한 결과 신호를 쓰면 별도 adaptation임을 명시한다. 종료 전용 기준은 기존 네 행동 head를 강제로 바꾼 점수와 구분해 독립 학습한다.

**Outcome CE도 미래 결과로 decision policy를 학습한다.** Teacher 행동 모방과 outcome supervision을 구분해야 하며, CE 대 RL이라는 이름으로 정책 여부를 나누지 않는다.

반사실 표본의 return은 `Q(reference policy, state, action)`이다. 학습 후 새 policy의 최적 Q 또는 불편향 on-policy policy gradient라고 부르지 않는다. [AggreVaTe/NRPI 및 후속 학습 노트](research/20261023/trajectory_related.md)가 이 구분을 정리한다. 단지 알고리즘 이름을 PPO/GRPO로 바꾸는 것은 개선이 아니다.

## 공정한 벤치마크 계약

- **모델/생성 고정:** 현재 Qwen3-4B-Instruct-2507, non-thinking, 같은 prompt·도구 schema·argument generator·최대 step/context. 새 Qwen 후보 교체는 별도 축이다.
- **데이터 예산:** unique question 수뿐 아니라 실제 branch episode·생성/input token·GPU 시간·실패·재시도까지 기록한다. 비용이 다른 알고리즘은 동일 데이터 비교와 동일 총 계산 비교를 별도로 보고한다.
- **학습/평가 분리:** 기존 train24/val8과 과거 평가 데이터의 노출 이력을 보존한다. pilot2의 frozen40 결과를 본 뒤 그40으로 새 구조·보상·threshold를 고르지 않는다. 후속 평가는 새 family-disjoint holdout을 실행 전에 고정한다.
- **기준군:** native, syntax/cache를 동등하게 최적화한 native, BC, 같은 outcome-data CE, 종료 전용, 단순 선형 정책. 모든 구현을 이미 갖췄다는 뜻은 아니다. Native 확률을 uniform placeholder로 대체해 calibration 비교하지 않는다.
- **평가:** 최종 F1/EM, 도구 호출, 생성 토큰, supplied-prefix 토큰, 전체 prefill/input, 오류와 cap 강제 종료. Paired difference와 uncertainty를 보고한다. 실제 latency 주장은 통제된 환경의 별도 측정이 필요하다.
- **채택:** 품질 손실로 줄어든 호출을 성공이라 부르지 않는다. 학습용 데이터에서 범위를 정한 뒤 고정 평가에서 quality–cost 우세 여부를 확인한다. 새로운 head가 linear/MLP보다 안정적으로 낫지 않으면 단순 구조를 유지한다.

## 구조와 확률에서 배울 점

[SanSi와 Hev 보충 노트](research/20261023/architecture_probability_related.md)는 위치·층·readout 설계와 확률 해석을 다룬다. LoRA를 학습하는 모델과 head-only 모델, 한 번 읽는 모델과 여러 번 반복하는 모델의 비용을 섞지 않는다.

우리 `action_probabilities`는 **선택 분포 π(a|s)**다. 원하는 경우 성공확률과 남은 비용을 별도 target으로 예측할 수 있지만, 높은 π가 높은 정답확률임을 뜻하지 않는다. Entropy를 높여 탐색이 늘어나는 것과 calibration이 좋아지는 것도 다르다. 현재 temperature 조정은 train-only 탐색 보강이며 정답확률 교정 성공 주장이 아니다.

## 자료와 확인 범위

- [Head·동일 backbone](research/20261023/head_related.md)
- [Trajectory·반사실·cost-to-go](research/20261023/trajectory_related.md)
- [효율적 검색·종료](research/20261023/efficient_rag_related.md)
- [구조·확률 보충](research/20261023/architecture_probability_related.md)
- [Source ledger 폴더](../lab/reports/related_work_20261023/): 논문별 URL/버전/방법/데이터/학습 대상/확인 한계.
- [진행 중 pilot2 protocol](JEV_DECISION_POLICY_PILOT2_20261023.md), [pilot1 결과](JEV_DECISION_POLICY_PILOT1_REPORT_20261023.md).

외부 논문 결과를 우리 환경에서 재현했다고 주장하지 않는다. 이번 산출물은 원문 비교·재현 가능성 점검·구체적 benchmark adaptation 설계다. 실행 결과는 각 실험의 별도 보고서에만 추가한다.
