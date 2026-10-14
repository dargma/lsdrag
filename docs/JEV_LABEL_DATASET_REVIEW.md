# jev_v1.0 라벨 자료 검토 — 필요한 수집만

2026-10-14. 사용자 제안은 약 1,000문항을 한 번씩 풀고 성공 궤적을 teacher label로 쓰는 것이다. Astra 재검토 후 **기존 코퍼스의 새 M3부터 시작하고, 부족할 때 MultiHop-RAG만 보완**하기로 최소 계획을 정리했다. 모델 추론은 아직 실행하지 않았다.

## 1. 실제 준비 규모

| 자료 | CPU로 확인한 수 | 의미 |
|---|---:|---|
| M3의 원 질문인 MMQA dev | 2,441 | 공식 M3DocVQA가 사용하는 질문 기반 |
| 현재 코퍼스에서 support 문서가 포함되는 문항 | 201 | 이미지 crop·내용 완전성 QC와는 별개 |
| 기존 100문항 제외 새 질문 | **102** | 이미지 근거 49·비이미지 53 |
| 새 최종 평가 / 학습 수집 | **20 / 82** | 평가: 이미지 10·비이미지 10, outcome 전 seed 고정 |
| 공식 MMQA train | 23,817 | 원본 metadata를 추가 확인 |
| MMQA train 중 현재 문서 URL로 덮이는 후보 | **37** | 이미지 근거 23. alias·실제 그림 QC 후에만 사용 |
| 별도 M3 1,000문항의 필요한 새 문서 | **1,364** | 기존 100 제외·seed 20261014 층화 추출의 준비 비용 추정 |

[M3 공식 생성 설명](https://raw.githubusercontent.com/bloomberg/m3docrag/main/m3docvqa/README.md) · [MMQA 공식 자료](https://github.com/allenai/multimodalqa) · [로컬 감사 JSON](../lab/reports/jev_v1.0/dataset_feasibility.json)

기존 100문항 중 Rudy Robbins 질문 1개는 필요한 표 문서가 현재 IR의 문서 별칭 목록에 없었다. 감사 JSON에 qid·누락 ID를 기록했다. 기존 예측·점수를 소급 변경하지 않고, 새 학습/QC에서 누락된 근거로 오답 라벨을 만드는 일을 피한다.

## 2. 대안 비교

| 데이터셋 | 현재 도구와의 적합성 | 이번 결정 |
|---|---|---|
| **준비된 M3 + QC 통과 MMQA train** | 같은 문서·이미지 환경과 가장 가까움 | **우선 사용** |
| **MultiHop-RAG** | 텍스트 다중 문서 QA. keyword/semantic/read로 같은 종류의 행동 궤적 생성 가능 | **유일한 fallback** |
| HotpotQA | 공식 train이 있고 검색·다중 근거 학습에 적합하나 별도 corpus·전처리 필요 | 이번에는 추가하지 않음 |
| ToolACE | 11,300개 공개 툴 호출 자료. 하지만 API·행동 공간이 우리 세 도구와 다르고 이미지가 없음 | 주 라벨로 사용하지 않음 |
| M3 1,000문항의 PDF 확대 | 멀티모달 환경은 맞지만 새 문서 1,364개 준비·파싱·인덱싱 필요 | 이번 최소 실험에서 제외 |

근거: [MultiHop-RAG 저자 저장소](https://github.com/yixuantt/MultiHop-RAG) · [HotpotQA 공식 자료](https://hotpotqa.github.io/) · [ToolACE 저자 공개 데이터](https://huggingface.co/datasets/Team-ACE/ToolACE).

## 3. MultiHop-RAG는 어떻게 쓸 것인가?

저자 공개 파일을 직접 읽어 **질문 2,556개·코퍼스 609개 기사**를 확인했다. question_type은 inference 816 / comparison 856 / temporal 583 / null 301이다. [원본 데이터](https://huggingface.co/datasets/yixuantt/MultiHopRAG)

- 원본 유형 비율에 따라 1,000개 행을 뽑고 seed로 순서를 고정했다. 결과를 본 뒤 쉬운 문항으로 바꾸지 않는다.
- 필요할 때만 100개씩 기존 Gemma A-RAG teacher로 풀며 성공 궤적을 모은다. 매 block마다 새 모델을 학습하지 않는다.
- 코퍼스는 전체 609개 기사이며 정답 evidence만 검색 대상으로 제공하지 않는다. `evidence_list`·정답은 채점에만 사용한다.
- null 유형은 별도 표시·채점하며 빈 답 성공만 늘려 데이터 충분성 기준을 채우지 않는다.
- 별도 MultiHop test 성능을 측정하는 실험을 추가하지 않는다. 목적은 학습 라벨 보완이다.
- 이미지 없는 데이터이므로 사용했다면 최종 결과를 **혼합 학습→M3 전이 PoC**로 명시한다.

문항 수보다 **서로 다른 성공 문항 수와 class coverage**가 수집 종료 기준이다. 최대 1,000개를 모두 돌린다는 뜻이 아니라 필요한 수를 확보하면 중단하는 상한이다.

## 4. 고정된 파일

[수집 계획](../lab/registry/jev_v1.0/label_collection_plan.json)에 train82/test20 qid, MMQA 후보 ID, MultiHop 최대 1,000개 원본 행 번호·실행 순서·원본 파일 해시가 있다. 원 질문·정답은 로컬 manifest에서 분리했다. 실행 Agent에는 question만 제공한다.

[최소 실험 계획](JEV_EXPERIMENT_PLAN.md)의 2-arm·40회 최종 비교 외 추가 benchmark·수집 sweep는 이번 범위가 아니다.
