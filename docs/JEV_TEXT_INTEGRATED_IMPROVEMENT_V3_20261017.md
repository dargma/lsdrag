# 행동 선택·read 인자 선택 통합 개선 실험 v3

2026-10-17 사용자 승인: 기존 개선계획v2의2·3단계를 통합한다. 이번 요청은 계획 저장·백업·Git push이며 실험 시작은 아니다. 자동 실행은 해제 상태를 유지한다. v2 원본과 기존 결과는 보존한다.

## 공통 설정

Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554, frozen backbone, greedy/non-thinking. MuSiQue-Ans2hop 텍스트 전용, keyword_search/semantic_search/read_chunk/FINAL. 모델·데이터·실행·Git은 local-fast, Drive는 백업. 생성 없는decision-only 별도 실험, RL, step압축, reader미세학습은 제외한다.

## 1. 생성 경로 진단 결과 검토

train8의24고정상태 실행은 완료. exact native prefix cached OFF 토큰24/24일치, grammar ON 값24/24일치, compact prefix 값17/24일치. 오류0이나 logits parity 기준 전체통과는 아님. 수치차이와 구현오류를 검토한 뒤 진행한다. 의미내용을 포함한 native prefix는oracle진단이며 배포시 강제하지 않는다. 기존24상태 결과는 새학습완료 증거가 아니다.

## 2. 행동 head와 read 인자 head 통합

고정 train24/validation8은 improvement_v2_split.json을 그대로 사용한다. 기존32문항은 개발에 노출되었으므로 독립heldout이라고 부르지 않는다.

- 단일 Qwen prefill hidden state → 4-way 행동head와5-way read-slot head.
- 행동head는keyword/semantic/read/FINAL. read-slot은GENERATE 또는기존관측ID묶음/단일ID후보최대4개. 없는후보는mask한다.
- 전체natural-state action CE + read상태만 read-slot CE(계수1). CPU,seed20261016,100epochs,AdamW lr0.001/wd0.01. train-only 정규화. backbone학습 없음.
- 검색인자는항상생성. read후보선택은즉시실행하고GENERATE이면같은결정KV cache에서생성. FINAL은공통Qwen생성. 별도인자선택LLM호출 없음.
- read label은현재후보와teacher인자가정확히일치하면해당slot,없으면GENERATE. gold관련성으로oracle선택하지않고 임의ID subset구조를추가하지않는다.
- 기존30qid gate는역사프로필에유지. train24용gate와checkpoint계약을구현전에고정하고 action/read선택/GENERATE별분모를보고한다. coverage부족을성공으로처리하지않는다.
- prompt/입력token계약이변하면feature재추출. 기존32전체로학습한head를validation baseline으로사용하지않는다.

필요시복구상태추가1회만허용한다. 고정train8에서최대16state,teacher후속상태당최대4tool calls+FINAL/1024completion tokens. 전체80reader calls/16384completion tokens/15GPU-worker분 중먼저도달하는한도. 후보checkpoint는기존상태/복구추가최대2개. validation고정상태행동 macro F1개선과premature FINAL오류증가없음을기준으로checkpoint하나를선택한다. 클래스부재는판정불가로기록한다.

선택한 **동일 checkpoint**로 validation8×2조건(최대16episode)을 비교한다.

| 조건 | 행동 선택 | read 인자 |
|---|---|---|
| always-generate | 같은action head | 항상생성 |
| 통합read-selection | 같은action head | 기존후보선택 또는GENERATE |

read선택실사용>0,F1/EM악화없음을채택기준으로하고생성량·prefill·forward·선택비용을함께기록한다. 효과없음/악화면always-generate유지. 두조건의자율trajectory는분기할수있으며동일step보장으로표현하지않는다. 반복튜닝/무제한retry없음.

## 3. 설정 동결 후 fresh20 평가

validation에서primary하나를선택하고코드/prompt/checkpoint/model/limits해시를동결한다. 모든기존train/노출dev를제외한fresh20을결과를보지않는hash순서로고정한다. native A-RAG대primary,20×2=40episode. 제3조건은평가전에등록할때만추가하여최대60episode. heldout결과로primary교체/반복튜닝하지않는다.

F1/EM/support,승패동률,빈검색·종료오류,실제tool calls,생성/forced-prefix tokens,prefill/forward/선택비용을보고한다. 작은표본의탐색결과이며비열등성이나공유GPU속도향상입증으로부르지않는다. 운영상중단은미완료로기록한다.

## 상태와 다음 행동

현재: 1단계24상태 실행완료/원인검토필요. 통합학습·validation·fresh20은미실행. 자동실행해제유지. 다음작업은1단계parity및생성경로검토후통합프로필구현이다.

기계판독원본: lab/registry/text_integrated_20261016/improvement_experiment_plan_v3.json. 이전Astra검토는v2에대한것이며v3새검토를받았다고표현하지않는다.
