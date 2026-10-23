# Memory-R1 조건부 후속 실험 계획 — 2026-10-23

## 상태와 최신 사용자 요청

사이드 대화에서 사용자가 계획서 저장·Git commit/push를 요청했다. 이 문서는 계획이며 runner/자동실행 설정을 변경하거나 실험을 기동하지 않는다. 메인 작업의 A-RAG 구현·평가·검증·백업을 우선한다.

사용자 최신 A-RAG 목표: 고정 GRPO 대비 F1 +3 percentage points, 평균 생성 토큰 −20% 동시 달성. 기존80문항의 설명용 수치는 F1 .3343263 → .3643263 이상, generated56.25 →45 이하이다. 실제 판정은 새로운 공통 평가셋의 동시 baseline 대비로 하며, 목표 평균 달성과 통계적 유의성을 분리한다. 이 문서는 이전 −15% 목표를 갱신하는 요청을 기록한다. 실행 중인 manifest/선택 코드에는 자동 반영되지 않으므로 메인 작업에서 기존 평가 노출 여부를 확인하고 적용 시점·변경을 기록해야 한다.

## 실행 조건과 시간

A-RAG 작업이 8시간 상한(기존 계획 2026-10-23 22:48 UTC) 전에 완료되고 결과·checkpoint·원본검증·백업이 확보되면 남은 시간에 Memory-R1 방식의 작은 관리 과제를 검토한다. A-RAG 목표 달성은 바람직하나 사용자가 후속 실행의 필수조건으로 명시한 것은 조기 완료이다. 목표 미달이면 원인과 전이 가능성을 보고, 메인 작업이 시간과 위험을 판단한다. 미완료 A-RAG를 중단하고 확장하지 않는다.

잔여시간으로 실제 준비·smoke 측정 후 완주가 가능한 규모만 실행한다. 최소 30분은 보고·백업에 예약한다. 잔여시간이 부족하면 데이터/schema/protocol 준비까지만 저장한다. 시간 부족으로 평가를 생략한 결과를 성공으로 표시하지 않는다. GPU worker는 하나만 사용하며 메인 작업 또는 다른 프로세스를 종료하지 않는다.

## 연구 질문과 범위

동일한 소형 LLM에서 메모리 관리 행동을 자연어/JSON으로 생성하는 방법보다, 공통 hidden을 읽는 작은 action head로 선택하는 방법이 이후 QA 품질을 유지·개선하면서 제어용 생성 토큰을 줄이는가?

Memory-R1 원문: https://arxiv.org/abs/2508.19828 . 아래 설정은 원문의 전체 재현이 아니라 관리 행동을 대상으로 한 단일 GPU adaptation이다. 원문 전체 결과와 수치를 직접 비교하지 않는다. A-RAG의 검색 행동과 memory 작업은 의미가 다르므로 출력 layer는 새로 초기화하고 새 행동으로 학습한다. backbone/입력 처리/학습 코드 재사용과 checkpoint의 zero-shot policy 전이를 구분한다.

## 최소 실행 과제

- 입력: 과거 메모리 snapshot + 새로운 정보 + 관련 메모리 후보.
- 행동: ADD / UPDATE / DELETE / NOOP.
- 첫 pilot은 관련 후보를 결정론적으로 하나로 제한한다. baseline과 head에 같은 후보를 준다. 대상 선택 능력은 입증하지 않으며, 여러 후보의 slot head는 첫 결과가 유망하고 시간이 남을 때 후속으로만 추가한다.
- ADD/UPDATE 내용은 동일한 고정 LLM으로 생성한다. DELETE/NOOP는 내용 생성 없이 실행한다.
- 공통 LLM과 head를 결합한 decision-generation 모듈이다. head 자체가 문장을 생성하는 것은 아니다.
- 답변 시 retrieval과 answer generator는 고정한다. 메모리 관리 시 미래 평가 질문·정답은 입력하지 않는다.
- 단일 변경 후 QA부터 시작한다. 긴 메모리 trajectory, 장기 망각 및 일반화는 이번 pilot의 주장 범위 밖이다.

## 데이터와 분할

우선 공개 대화 기반 데이터에서 정보 추가·중복·정정·명시적 무효화 상황을 포함한 작은 snapshot 집합을 구성한다. 공개 출처·라이선스·revision·생성 규칙·hash를 기록하고 local-fast에 둔다. 정확한 데이터 가용성은 아직 확인하지 않았으므로 확보 완료로 표시하지 않는다.

희망 규모: train32 / validation8 / test24 snapshots, QA는 snapshot당 새 사실 질문과 보존해야 할 기존 사실 질문 각1개. 가능한 경우 대화/사용자 단위로 분할하고 같은 대화의 파생 snapshot을 다른 split에 넣지 않는다. 데이터가 이 분할을 지원하지 않으면 규모를 줄여 탐색 pilot으로 표시한다. 합성 fixture는 실행 smoke용이며 실제 데이터 품질 평가와 합산하지 않는다.

## 비교와 비용 공유

필수 비교는 세 조건이다.

1. 고정 LLM이 행동을 생성하는 관리 baseline.
2. 같은 baseline의 train 행동으로 초기화한 action head(BC).
3. 같은 초기 head를 행동별 이후 QA 결과로 개선한 head.

모든 조건의 모델·초기 메모리·관련 후보·내용 생성 설정·QA 경로·메모리 크기 제한을 일치시킨다. 학습 데이터와 결과 수집 비용을 별도 보고하며 baseline이 무학습인 차이를 숨기지 않는다.

한 train snapshot에서 합법 행동을 모두 적용해 결과를 평가하고 hidden/후속 QA를 캐시한다. 동일 행동의 후보 내용은 고정 decoder로 한 번 생성해 학습 분기에서 공유한다. 실제 test에서는 각 방법의 선택부터 끝까지 실행하고 비용을 다시 측정한다. gold answer로 저장 내용을 생성하지 않는다.

## 학습 및 확률 활용

첫 pilot은 같은 상태의 모든 합법 행동 결과를 사용하는 regularized expected-utility policy improvement를 우선한다. 강제 실행한 행동 결과를 on-policy GRPO로 오인하지 않는다.

Q(s,a) = mean downstream QA score − lambda × normalized management cost.

J(theta) = sum_a pi_theta(a|s) Q(s,a) − beta KL(pi_theta || pi_initial).

행동 선택확률 전체로 기대효용을 계산한다. 이 확률을 QA 정답 확률이라고 해석하지 않는다. QA 점수에는 신규 정보 활용과 기존 정보 보존을 함께 반영한다. 비용 정규화와 lambda는 train/validation에서만 결정해 test 전에 동결한다. 별도 learned judge는 초기 pilot에서 사용하지 않고 공개 정답 기반 EM/F1과 schema/실행 검증을 사용한다. 별도 내용 의미평가가 없다는 한계는 기록한다.

train32 × 최대4행동 × QA2 = 최대256 QA 평가를 기본으로 한다. validation8/test24 ×3조건 × QA2 =192 QA 평가. 합계 최대448 QA 평가 외에도 메모리 내용 생성·baseline 행동 생성·초기화 teacher 비용을 반드시 합산한다. 반복·대규모 GRPO는 기본 계획에 포함하지 않는다. smoke 처리량으로 episode/token/wall 상한을 runner manifest에 확정한 뒤 실행한다.

## 지표와 판정

주 지표: 동일 test snapshot의 paired downstream QA F1, 전체 생성 토큰. 관리용 생성·답변 생성·입력 토큰·memory token·호출·schema 오류를 분리한다. 상태를 한 쌍으로 bootstrap하며 같은 snapshot의 두 QA를 독립 표본처럼 세지 않는다.

우선 효용 판단은 생성 기반 선택 대비 QA 성과와 비용의 tradeoff이다. A-RAG의 +3%p/−20% 목표를 다른 과제에 자동 적용하지 않는다. test24는 제한적인 feasibility 결과이며 비열등성·일반화 입증으로 해석하지 않는다. 토큰 절감이 단지 행동명 JSON을 생략한 결과인지, 더 좋은 메모리 선택까지 동반한 것인지 BC 대 결과학습 비교로 구분한다.

## 동적 선택과 종료

- schema/상태변경 오류가 있으면 학습 확대 전에 수정하고 smoke를 재검증한다.
- 대부분 행동이 같은 QA 결과를 내면 모델을 키우지 않고 train의 중복·정정·보존 coverage를 보완한다.
- 행동별 결과 차이가 충분한데 head가 일반화하지 못하면 A-RAG에서 선정된 입력/구조 하나만 재사용한다. 새로운 구조 조합 탐색은 하지 않는다.
- 시간이 부족하면 scope를 줄이고 완료한 단계만 명시한다. test를 본 뒤 target/lambda/후보를 고르지 않는다.
- 최종 결과·실패·분할·코드/모델 hash·비용을 보고서에 기록, 큰 원본/weights는 Drive archive, 작은 문서·provenance만 jev_v1.2에 commit/push.

관련 A-RAG 계획: [8시간 개선 계획](JEV_DECISION_EIGHT_HOUR_PLAN_20261023.md).
