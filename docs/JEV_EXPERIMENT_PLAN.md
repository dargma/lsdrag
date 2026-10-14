# jev_v1.0 최소 실험 계획 — Gemma Reader의 decision head

2026-10-14 · **Astra 재검토 반영본** · 브랜치 `jev_v1.0`.

**이번에는 두 조건만 비교한다: 원래 Gemma A-RAG vs 같은 Gemma에 행동 head를 붙인 A-RAG.** 데이터 수집도 하나의 경로와 하나의 보완책으로 제한한다. 현재 실행 현황은 [진행 보고](../lab/reports/jev_v1.0/PROGRESS.md)에 기록한다. teacher 수집·head 학습·20문항×2 비교를 완료했다. 최종 수치는 진행 보고에 있으며 20문항 PoC의 한계를 함께 기록했다.

## 1. 목표와 고정 조건

목표는 중간 턴의 툴 선택을 텍스트 생성에서 분류로 분리했을 때 **최종 답변 정확도와 전체 풀이 시간이 어떻게 바뀌는지** 확인하는 것이다.

| 항목 | 고정 조건 |
|---|---|
| 모델 | 현재 로컬 **Gemma 4 E4B-it**, 같은 고정 revision |
| 변경 | backbone 동결, 마지막 문맥 표현에 작은 **4-class linear head**만 학습 |
| 행동 | keyword_search / semantic_search / read_chunk / FINAL |
| 생성 | 선택 뒤 필요한 검색어·청크 ID 인자와 최종 답변은 같은 Gemma가 생성 |
| M3 검색 | 기존 EmbeddingGemma 2 텍스트+이미지 검색·인덱스·RRF 유지 |
| Reader 입력 | 같은 질문·history·tool definitions·검색된 실제 이미지 |
| 예산 | A-RAG max iteration 8 + 별도 강제 최종 호출, 이미지 최대 8개·soft tokens 1120 |
| 외부 서비스 | **OpenAI·Jev·외부 teacher API 사용하지 않음** |

Jev·Laya·Kev·SemIf는 [학습법 조사](JEV_DECISION_MODEL_TRAINING_REVIEW.md)의 참고 자료다. 다른 모델로 Reader를 교체하거나 Jev의 비공개 RLCD를 재현하는 실험이 아니다.

## 2. 데이터 계획 — 1,000문항은 최대 보완 예산

### 이미 확인한 문제

기존 M3 100문항 × 3조건의 300궤적·955 Reader turns에서 성공은 9개였다. 모든 툴 호출은 단독 호출이었다. outcome과 무관하게 고정한 기존 60/20/20 분할의 train 성공 라벨은 **4문항·5궤적·15턴, read_chunk 양성 0**이다. 따라서 이 라벨만으로 4-class 일반 평가를 시작하지 않는다. [기존 감사](../lab/reports/jev_v1.0/teacher_audit.json)

현재 288문서에서 필요한 근거 문서가 포함되는 새 M3 질문은 **102개**다. 기존 100개를 제외한 M3 후보에서 1,000개를 층화 추출하면 **새 근거 문서 1,364개**가 필요했다. 이를 모두 PDF로 다시 준비하는 작업은 이번 최소 실험에서 제외한다. [준비 가능성 감사](../lab/reports/jev_v1.0/dataset_feasibility.json)

### 실행할 단일 경로

| 순서 | 자료 | 수행 |
|---|---|---|
| 먼저 잠금 | 새 M3 **20문항**: 이미지 근거 10·비이미지 근거 10 | 결과를 보기 전에 최종 평가용으로 고정. 학습·수집 중 열지 않음 |
| 1차 수집 | 나머지 새 M3 **82문항** | 기존 A-RAG teacher로 문항당 한 번 풀고 성공 궤적 수확 |
| 저비용 보강 | 공식 MMQA train 중 현재 문서 URL로 덮이는 **37개 후보** | IR 별칭·실제 이미지 매핑 QC를 통과한 문항만 같은 방식으로 추가 |
| **유일한 대안** | **MultiHop-RAG 최대 1,000문항**, 전체 코퍼스 609개 기사 | 앞의 자료로 라벨이 부족한 경우에만 사용. 미리 고정한 순서로 100문항씩 수집하고 충족 시 중단 |

[frozen collection plan](../lab/registry/jev_v1.0/label_collection_plan.json)에 새 M3 train/test ID와 MultiHop-RAG 원본 행 번호 1,000개·해시·실행 순서를 저장했다. 기존 `split.json`은 보존하며, 이전 test 20문항을 다시 측정하거나 train으로 옮기지 않는다. 기존 train 60의 성공 궤적은 재실행 없이 사용할 수 있다. 기존 development 20은 모델 선택용 자료로만 유지한다.

### 라벨 충분성·중단 규칙

- 최소 진입 기준은 **답할 수 있는 학습 질문의 서로 다른 성공 문항 30개 이상**, 그리고 네 행동 각각이 **10개 이상의 서로 다른 성공 문항**에서 관측되는 것이다. PoC 최소 기준이며 통계적 충분성을 보장하지 않는다.
- 100개 block마다 **라벨 개수만** 확인한다. 매번 head를 재학습하거나 test를 보지 않는다. 학습은 수집 종료 후 한 번의 고정 절차로 수행한다.
- MultiHop-RAG의 답할 수 없는 질문은 원본 유형을 보존해 별도 집계한다. 원본 정답이 `Insufficient information.`인 문항의 abstention 채점 규칙은 실행 전에 고정한다. 빈 답으로 맞힌 문항만 늘려 30개 성공 기준을 통과시키지 않는다.
- 최대 예산에도 class가 부족하면 ‘자료 부족’으로 보고하고 폐루프 비교로 넘어가지 않는다. 자동으로 다른 dataset·더 큰 모델·재시도를 추가하지 않는다.
- 라벨은 성공한 teacher의 **선택 모방**이며 최적 행동이나 답변 정답 확률이 아니다. 정답·support metadata는 채점·QC에만 쓰고 teacher 입력에 제공하지 않는다.

MultiHop-RAG는 이미지 없는 텍스트 다중 문서 QA다. 보완에 사용했다면 결과는 **혼합 자료로 학습한 head의 M3 전이 PoC**로 명시한다. 여러 dataset을 합친 검증 성능으로 보고하지 않는다. MMQA의 URL 포함 여부도 실제 이미지 일치·PDF 내용 완전성을 보장하지 않아 QC가 필요하다. [자료 비교](JEV_LABEL_DATASET_REVIEW.md)

## 3. 구현은 필요한 것만

### 같은 prefill에서 head 읽기

전체 Gemma 멀티모달 forward를 유지하고 기존 generation boundary의 마지막 정규화 hidden vector만 읽는다.

$$
\mathbf{z}=W\mathbf{h}+\mathbf{b},
\qquad
p(a\mid s)=\operatorname{softmax}(\mathbf{z})_a
$$

$$
\mathcal{L}_{\mathrm{CE}}=-\log p(a_{\mathrm{teacher}}\mid s)
$$

head만 CE로 학습한다. seed와 학습 설정을 고정한다. 이번 최소 실행은 full-batch AdamW 100 epochs, lr 0.001, weight decay 0.01, plain CE 한 번이며 development/test sweep은 하지 않는다. 현재 자료량으로 calibrated confidence·학습된 fallback threshold를 주장하지 않는다. 범위 밖 행동·잘못된 인자 등 구조적 오류는 기존 생성 경로로 돌아가며 그 비용을 센다. original Agent의 복수 툴 호출 기능은 없애지 않지만 head가 복수 행동을 학습했다고 주장하지 않는다.

### 인자 생성과 이미지 입력 보존

head가 semantic을 골라도 검색어는 필요하다. 같은 KV에서 선택된 툴의 native prefix 뒤 인자를 생성한다. FINAL은 기존 답변 JSON 형식의 `{"answers":` prefix 뒤 답을 생성한다. 단순히 tool 토큰만 막으면 툴을 일반 텍스트로 흉내 낼 수 있어 JSON prefix·형식 검증을 사용한다. prefix 비용도 측정한다. 이 prefix 처리·인자 생성·fallback 시간까지 포함한다. KV 재사용이 안전하게 확인되지 않으면 이중 prefill을 허용하되 실제 두 번의 비용을 기록한다.

vision·PLE 처리를 우회하지 않는다. `output_hidden_states=True`로 42층 전체 표현을 저장하지 않고 마지막 표현만 읽는다. Gemma tool template을 수작업으로 가정하지 않으며 first turn·tool response·이미지 추가·강제 최종 호출의 경계를 확인한다.

필수 사전 검증은 **train에서 고른 text state 1개·image state 1개**의 실행 확인과 parser·이미지·최종 호출 테스트다. 별도 대규모 smoke benchmark를 만들지 않는다.

## 4. 최종 비교 — 정확히 두 조건, 40회 풀이

| 조건 | backend·모델 | 차이 |
|---|---|---|
| **B-HF** | 같은 HF·같은 Gemma | 기존 native 툴 호출 생성 |
| **D-head** | 같은 HF·같은 Gemma | head로 행동 선택, 인자·답변은 생성 |

새로 잠근 M3 **20문항 × 두 조건 = 40회**를 문항·조건별 한 번씩 실행한다. 조건 순서는 seed로 교차 배치하고 동일 GPU에서 순차 실행한다. 오답 재시도·best-of·정답 근거 직접 제공은 없다.

**필수 결과만 보고한다.**

- **답변:** EM·F1, 문항별 B-HF/D-head 답과 성공↔실패 차이.
- **시간:** 전체 wall time과 중앙값, prefill·head·인자·검색·최종 답변·fallback 비용. 준비·학습 비용은 별도.
- **동작 확인:** 툴/인자 오류, 모델 호출 수·토큰 수·fallback 수, 실제 이미지 입력 보존.
- **제약:** class별 학습 문항 수, text/image 구성, QC 탈락, GPU peak memory·다른 세션 경합, 소규모 표본의 불확실성.

이미지 전달 recall은 정답 이미지 대응을 검증한 문항에 한해 보조적으로 낼 수 있지만, 새 20문항 전체의 이미지 정렬을 다시 만드는 작업을 이번 주 평가의 필수 조건으로 늘리지 않는다. 오프라인 teacher 행동 일치율을 최종 정답률 대신 쓰지 않는다. 표본 20개로 p95·비열등성·calibration 안정성을 주장하지 않는다.

시간 계산에서 인자와 fallback을 제외하지 않는다.

$$
\begin{aligned}
T_{\mathrm{total}}&=T_{\mathrm{prefill}}+T_{\mathrm{decision}}+T_{\mathrm{arguments}}\\
&\quad+T_{\mathrm{tools}}+T_{\mathrm{final}}+T_{\mathrm{fallback}}
\end{aligned}
$$

## 5. 이번에 하지 않는 실험

**무학습 logits 비교군, vLLM 재평가, LoRA, RL, calibration sweep, 추가 test benchmark, branching·memory·context 제어 실험은 실행하지 않는다.** 최신 조사와 [contribution 가설](JEV_CONTRIBUTIONS.md)은 후속 연구 배경으로 보존한다. 먼저 두 조건에서 동작·시간·정확도를 확인한 뒤 새로운 근거가 있을 때 다음 단계를 정한다.

공유 GPU의 다른 세션을 종료하지 않는다. 초기 우리 모델의 목표 상주량은 25GB 이하이며 실제 peak와 여유를 확인한다. 메모리 부족·충돌 위험은 알리고 새 GPU 단계를 시작하지 않는다. 새 1,364문서 파싱과 인덱스 확대도 이번 범위에서 제외한다.

## 6. Astra 재검토 결론과 현재 상태

Astra의 두 번째 검토는 **두 비교군 + 새 M3 20 holdout + 한 가지 데이터 fallback**으로 압축하는 안에 동의했다. 필요 검증은 데이터/이미지 QC, 최소 text/image 실행, 인자까지 포함한 실제 시간과 답변 품질이다.

현재: teacher 수집·HF text/image 검증·head 학습·20문항×2 비교 완료. [진행 보고](../lab/reports/jev_v1.0/PROGRESS.md). 브랜치·자료 조사·기존 감사·새 분할·수집 순서·문서 검증 완료. [train82 기본 입력 QC](../lab/reports/jev_v1.0/train82_input_qc.json)에서 82개 질문의 입력 필드·support 문서와 이미지 필요 39개 문항의 추출 이미지 파일 존재를 확인했다. 정답 그림 crop의 의미 일치 검증과는 별개다. 새 M3 82·MMQA 37·MultiHop-RAG 300문항 수집 후 성공 문항 45개로 라벨 기준을 충족했다. 197턴에서 head를 한 번 학습하고 평가 40회를 완료했다. GPU 사용을 종료했다.

참조: [label 계획](../lab/registry/jev_v1.0/label_collection_plan.json) · [학습법](JEV_DECISION_MODEL_TRAINING_REVIEW.md) · [Astra RAG·memory 조사](JEV_MEMORY_RESEARCH_ASTRA.md) · [기존 Agent](../src/lsdrag/agent.py).
