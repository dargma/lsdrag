# jev_v1.1 실행 직전 재검토

작성: 2026-10-15 UTC. 범위: 기존 제안과 Astra 검토를 실제 1·2단계 실행으로 좁히는 감사. 이 문서 작성에서는 GPU 실행·설치·git 변경을 하지 않았다. 새 결과 또는 방법의 성공을 주장하지 않는다. 기준 문서는 [제안](JEV_ICML2027_RESEARCH_PROPOSAL.md), [Astra 검토](JEV_ICML2027_ASTRA_REVIEW.md)다.

## 판정

**1단계 실행 승인. 2단계는 구현·작은 학습까지 준비하되, 1단계의 데이터 유효성과 Reader 신호를 확인한 뒤 같은 작업 흐름에서 순차 실행한다.** 사용자가 이미 두 단계의 진행을 요청했으므로 단계 사이의 추가 사용자 승인은 필요 없다. 통과하지 못한 gate를 무시하고 학습 결과를 만드는 것은 완료가 아니다. 그 경우 원인을 기록하고 미리 정한 한 번의 수정/재진단 후 중단한다.

지금은 새 데이터셋·새 Reader·새 임베더·새 controller를 한꺼번에 교체하지 않는다. 기존 pinned Gemma와 아래 추가 검증된 MMQA 원자료로 Reader/timing을 확인하고, 모듈 경계가 분리된 작은 acquisition 실험을 실시한다. 복구 M3는 대체 경로로 둔다. 30–50문항으로 작은 head가 학습된다는 사실은 확인할 수 있지만, source 일반화나 ICML 기여를 검증했다고 할 수 없다.

## 1. 데이터 선택: CrossModalQA 보류, 공식 MMQA 원자료 우선

[CrossModalQA 원문](https://arxiv.org/html/2609.05518v1)은 1,863 QA, 4,987 articles, 4,431 images와 다섯 추론 경로를 기술한다. 그러나 이번 확인에서 저자 제공 다운로드·공식 GitHub/Hugging Face 데이터 주소를 확보하지 못했다. 원문 HTML에서 GitHub 링크는 arXiv 렌더링 지원 링크였고 Hugging Face 링크는 발견하지 못했다. 이름 및 저자 기반 GitHub/Hugging Face 검색도 공식 배포를 입증하지 못했다. **공개되지 않았다는 단정이 아니라 현재 실행 자산으로 검증되지 않았다는 판정**이다. 원문의 수치·샘플을 재구성해 새 데이터셋을 만드는 일은 이번 범위 밖이다.

M3는 [공식 M3DocRAG 저장소](https://github.com/bloomberg/m3docrag)와 [MultiModalQA 원자료](https://github.com/allenai/multimodalqa), 저장소의 pinned scorer 및 선택 manifest가 있다. 실제로 `/content/drive/MyDrive/lsdrag-backups/20261013/preflight`에 `mmqa`, `manifests`, `gold_images`, `pdfs` 디렉터리가 존재함을 확인했다. 파일 전체 무결성과 이미지 연결은 실행 담당의 복구 검증 대상이다.

최소 데이터 경로:

1. 기존 development 후보에서 결과와 무관하게 고정한 30–50문항을 Reader/timing 진단에 쓴다. 질문·정답·gold supporting text/table/image를 보존하고 이미지 누락을 별도로 센다.
2. 먼저 `question only`, `gold evidence only` 조건을 같은 최종 출력 형식으로 비교한다. gold answer·gold intermediate entity·gold query·chain 설명은 Reader 입력에서 제외한다. 이미지 원본 + 텍스트/표를 전달한 진단과 PDF page screenshot만 제공한 진단은 구분한다.
3. 2×2 실험의 e와 D는 **자연 검색 결과**여야 한다. gold support를 e로 선택하거나 정답 문서만으로 문항별 corpus를 좁힌 결과를 실검색 현상으로 보고하지 않는다.
4. 복구된 작은 공통 corpus를 쓰면 그 범위와 distractor 정책을 명시한다. 전체 benchmark 성능이라고 부르지 않는다. 이미지+표도 실험에 포함할 수 있으나 TEXT 행동에 table을 포함한다는 정의를 고정한다.

기존 M3 EM이 낮으므로 M3를 주 연구 데이터로 확정하는 것은 보류한다. 이번 선택은 빠른 복구와 bottleneck 진단을 위한 것이다. CrossModalQA는 QA·corpus·이미지·support ID·이용 조건·split을 실제로 확인한 뒤에만 후속 후보로 승격한다.

### 실행 담당의 신규 MMQA 확보 결과에 따른 우선 경로

위 M3 경로는 이 검토의 초기 fallback이었다. 이후 데이터 담당이 공식 MMQA train/dev 및 이미지 archive(약 2.36GB)의 접근과 range retrieval을 확인했고, text+image 질문에서 source component 기준 300 train / 80 dev 후보를 구성 중이라는 보고를 받았다. 이 접근 검증은 데이터 담당의 실행 결과이며 이 문서 작성자가 파일 전체를 별도로 다운로드·검증한 것은 아니다. **이 조건이 manifest와 파일 hash로 확인되면 기존 M3 복구보다 MMQA 원자료 우선을 승인한다.** PDF parsing/figure alignment라는 별도 실패 요인을 줄이고 원본 텍스트·이미지 획득의 접근 효과를 직접 확인할 수 있기 때문이다. 공식 corpus를 사용하더라도 새 reduced-corpus MMQA 실험을 M3DocVQA benchmark 결과라고 부르지 않는다.

먼저 고정된 40문항으로 stage1을 실행한다. 300/80의 전체 후보·split은 결과 전에 고정하되 전체를 즉시 branch하지 않는다. Reader가 풀 수 있고 자연 e에서 신호가 보이면 같은 규칙으로 train branch 수집을 확대한다. 원자료 schema, 이미지와 document ID 연결, list answer scorer 및 text-only와 image input이 실제로 서로 다른지 확인한다.

공통 corpus를 `선택한 모든 질문의 support union + 사전 고정 random distractors`로 만드는 것은 통제된 reduced-corpus 진단으로 허용한다. 문항별 gold scope는 금지한다. support union으로 구성한 사실과 distractor 수/비율을 공개하고 자연 전체 corpus prevalence나 open-domain 검색 난이도로 일반화하지 않는다. corpus를 구성할 때 사용한 gold 연결은 controller feature나 query/answer prompt에 절대 넣지 않는다.

source-component split은 동일 source에서 생성된 질문/답 supervision의 누수를 줄이지만, **공유 corpus에서 train rollout이 dev source를 실제 검색할 수 있다**. 따라서 support metadata만으로 만든 component split을 엄격한 unseen-source 평가라고 부르면 안 된다. 최소한 실제 train 입력의 source IDs와 dev support source IDs의 교집합을 감사한다. 같은 corpus에서의 질문/답 supervision source-disjoint 평가로 명명하는 것은 가능하다. 진정한 unseen-source 일반화는 별도 corpus partition 또는 노출 차단 설계가 필요하며 이번 실행을 위해 무리하게 추가하지 않는다.

## 2. 먼저 고쳐야 할 acquisition 및 입력 경계

### 행동 이름과 실행 kernel의 불일치

현재 [retrieval.py](../src/lsdrag/retrieval.py)의 `semantic_search`는 `V1/V2/G1`에서 text ranking과 image ranking을 RRF로 합친다. 이를 `TEXT_ACQUIRE`로 이름만 바꾸면 텍스트/이미지 행동의 의미가 깨진다. `image_search`는 `V2`에서만 노출되고 caption/snippet·figure ID를 반환한다. 그것만 전달하면 이미지 관측이 아니다.

실험용 kernel을 명시적으로 고정한다.

- `TEXT_ACQUIRE`: 공통 query generator → text-only index → 정해진 top-k 원문 text/table; 이미지 자동 첨부 없음.
- `IMAGE_ACQUIRE`: 동일 원칙의 query generator → image index → 정해진 top-k **이미지 bytes**와 사전 고정 metadata policy. 검색 설명만 주는 경로는 금지.
- `ANSWER_AND_STOP`: 공통 final-answer template와 parser. 생성형/one-token/head 모두 같은 answer format.

검색 모드별 corpus, embedding model/revision, query instruction, top-k, de-duplication, content truncation, image soft tokens 및 전달 규칙을 manifest에 고정한다. modality action과 retrieval-arm `B0/C1/G1/V2`를 같은 변수로 다루지 않는다. 매 호출 manifest에서 controller mode가 kernel/prompt를 바꾸지 않았는지 검증한다. lexical fallback을 쓰면 허용 조건과 횟수를 기록하며 다른 embedding kernel과 같은 실험으로 합치지 않는다.

### 모듈 접근 개입을 무효화하는 경로

`B`는 e를 얻기 전 공통 관측이고 분기 시점은 e를 받은 직후, 그 내용을 assistant가 다시 쓰기 전이다. `S0`에서도 acquisition 횟수·남은 budget·action mask는 `S1`과 같다. e를 숨겼다고 비용을 되돌려 주면 안 된다.

Query generator에 `S_j`를 주어 `D_j`를 실제 검색한다. final reader에는 `B`, 허용된 e, 실제 원문 D만 준다. query 문자열, query-generating assistant output, CoT, generated answer/summary, query echo wrapper를 넣지 않는다. 파일명·URL·title·caption·OCR에 e의 정보가 남으면 함께 기록하고 개입 단위를 image-only인지 evidence-package인지 정확히 명명한다. final reader에서 e를 제거했지만 앞선 assistant 메시지에 내용이 남아 있으면 해당 개입은 실패다.

`D0=D1`이고 reader 입력 bytes/token/image hash가 같으면 셀을 공유/캐시할 수 있다. 다른 점수라면 효과로 세기 전에 실행 비일관성을 조사한다. D가 e를 원문으로 재검색하는 것은 허용하되 재획득률을 보고한다. 이 결과는 순수 causal mediation의 증명이 아니다.

학습과 배포에서 hidden을 뽑는 **prefix가 같아야 한다**. answer 생성용 prefix로 학습한 h를 controller/tool schema prefix에서 읽어 쓰면 표현 분포가 바뀐다. controller observation builder를 하나 두고 학습 cache 및 실평가에서 함께 사용한다. query와 answer의 분리된 history를 도입했다면 모든 controller도 같은 경계를 사용한다.

## 3. 최소 estimand와 loss

각 e boundary와 각 acquisition a에서 아래만 수집하면 충분하다. 첫 pilot은 boundary 한 개, tool 두 개, 후속 acquisition 한 번 뒤 바로 답하는 horizon으로 고정한다. 이것은 first-action 장기 계획이 아니라 **관측된 evidence 이후 continuation**의 pilot이다.

```text
D_j(a) = retrieve_a(query_a(B + [e if j=1]))
U_ij(a) = score(answer(B + [e if i=1] + D_j(a)))
V_i = score(answer(B + [e if i=1]))

N0(a) = U01(a) - U00(a)
I0(a) = U10(a) - U00(a)
J(a)  = U11(a) - U10(a) - U01(a) + U00(a)
N1(a) = U11(a) - U10(a)
I1(a) = U11(a) - U01(a)

y_i(a) = U_ii(a) - lambda * future_cost_i(a)
y_i(STOP) = V_i - lambda * future_cost_i(STOP)
```

`future_cost`는 현재 이후 query generation, retrieval, image 처리, answer prefill/decode를 포함한 초 단위 비용이다. sunk cost는 빼고 budget에는 반영한다. utility는 pinned MMQA list-F1과 list-EM을 모두 저장하되 학습에 사용할 하나를 결과 전 고정한다. EM 저신호가 예상되므로 첫 회귀 실험의 primary target은 list-F1로 정하고 EM은 함께 보고하는 안이 합리적이다. 임의 substring 정확도로 교체하지 않는다.

첫 구현은 동일 h+budget metadata → 동일 작은 MLP trunk → 3-action Q 및 auxiliary heads로 충분하다. evidence pooling은 첫 실험에 필수 아님. frozen backbone과 같은 입력을 모든 학습 대조군에 제공한다.

```text
L_return = mean Huber(Q(S_i,a), y_i(a))
L_access = mean [Huber(n(S1,e,a), N0(a)) + Huber(j(S1,e,a), J(a))]
L = L_return + beta * L_access
```

첫 실행은 기존 검토의 `L_pair` 가중치 alpha=0을 권한다. return target의 차분을 다시 맞추는 loss는 새 정보를 주지 않으며 작은 표본에서 추가 튜닝 축이 된다. beta=0 및 고정 beta=1부터 비교하고 loss를 각각 유효 target 수로 정규화한다. lambda는 timing calibration 뒤, 본 효과를 보기 전에 작은 고정 grid로 정한다. lambda=0 결과도 남겨 QA 신호와 비용 효과를 구분한다. dev에서 본 값을 바탕으로 추가 grid를 넓히면 탐색으로 기록한다.

**off-diagonal U01/U10을 정상 Q-return label로 넣지 않는다.** S0/S1의 자연 kernel return은 U00/U11이다. 또한 e 제거 상태는 intervention augmentation이므로 자연 정책 평가와 별도 표기한다. 모든 baseline에 같은 S0/S1 factual returns를 제공한다.

최소 대조는 (a) 같은 구조의 return-only beta=0, (b) 동일 구조·학습 횟수의 return+access, (c) query-only와 고정 budget policy다. shuffled auxiliary target은 source/질문 단위 train 안에서만 섞어 비교할 수 있다. 같은 param count만 맞추려고 무의미하게 실제 inference 계산을 추가할 필요는 없지만 deployed path의 실제 비용을 보고한다. 생성형/one-token은 동일 kernel의 runtime 및 closed-loop 기준으로 유지한다.

이 최소 비교가 통과해도 수집 비용 동등 baseline은 아직 남는다. off-diagonal 비용으로 자연 branch를 더 수집한 generic value baseline까지 통과하기 전에는 학습 신호의 특수한 효용을 확정하지 않는다.

## 4. split과 평가 누수: 기존 manifest 재사용 금지

`lab/registry/jev_v1.0/split.json`은 qid shuffle이며 source-disjoint를 보장하지 않는다. `search_linked100_selection.json`은 이미 평가한 development50+evaluation50 합집합이고 `new_questions=0`이다. 이를 v1.1의 미관측 test로 다시 이름 붙일 수 없다.

- 이번 30–50 진단 및 기존 100문항은 pilot/development로 선언한다. 과거 결과를 본 사실을 숨기지 않는다.
- 학습용 실험을 하려면 원문 source/document/URL 및 가능한 entity의 연결 성분을 먼저 만들고 train/dev를 고정한다. 질문의 모든 e boundary, S0/S1, 2×2 셀, acquisition/action label, decoder trial을 같은 split에 둔다.
- giant component 때문에 source-disjoint split이 불가능하면 qid-held-out pilot이라고 명명하고 source 일반화 주장을 철회한다. 임의로 component를 잘라서 독립 표본처럼 만들지 않는다.
- train 평균/표준편차로만 feature scaling과 target preprocessing을 적합한다. auxiliary shuffle, early stopping, beta/lambda 선택에도 dev를 train로 끌어오지 않는다.
- 현재 성공 여부·gold modality·support ID·answer length·future query·U값·intervention index를 controller feature로 넣지 않는다. corpus ID는 필요하면 opaque ID로 쓰고 gold relevance를 표시하지 않는다.
- 평가 uncertainty의 단위는 질문 또는 source component다. 여러 branch를 독립 질문으로 세지 않는다. greedy decoder 반복은 표본 수를 늘리지 않는다.

## 5. 순차 실행과 gate

| 단계 | 산출물 | 다음 단계 조건 |
|---|---|---|
| 0. 복구 | Gemma `ee0ef6023621cff504d758262d4e04895a5af4a2`, processor, corpus/source hash, 실제 image delivery smoke | 같은 모델·prompt·이미지로 실행 가능; 별도 vLLM 설치는 필수 조건 아님 |
| 1A. 30–50 Reader/timing | question-only/gold-evidence paired answers, EM/F1, 형식 실패, end-to-end wall/peak memory | 이미지·정답 누수 없음, 유효 scorer; gold evidence에서 실제 해결 가능한 사례가 확보됨 |
| 1B. 작은 자연 2×2 | 실패 포함 사전 고정 boundary, U/V/cost, input hashes, N/I/J, 재노출 진단 | 동일 입력 불일치 0; query echo 없음; 최소한 학습할 nonzero contrast 및 행동 preference 변화가 존재 |
| 2A. cached head | return-only와 access-aux의 동일 split/architecture/optimizer 실행, checkpoint/config/hash | target 및 split 검증; 상수 target이면 degenerate label로 보고하고 성능 주장 중단 |
| 2B. dev closed-loop | 동일 kernel의 baseline/head QA–전체지연, action counts, fallback/timeouts | offline regression만 좋아진 결과와 실제 정책 결과 구분; 현상·속도·일반화 결론은 표본 한계 내에서만 |

수치 gate는 결과를 보기 전에 실행 manifest에 확정한다. 이번 작은 실행을 위한 보수적 운영 권고는 다음과 같다.

1. 30–50문항 중 gold-evidence에서 맞는 질문이 10개도 없거나 question-only 대비 순개선이 거의 없으면 dataset/reader bottleneck으로 처리한다. 이 수치는 유의성 검정이 아니라 작은 head에 필요한 신호의 최소 운영 기준이다. 잘못된 이미지·truncation·format을 한 번 수정한 뒤 재진단하며 개선된 질문만 골라 population을 바꾸지 않는다.
2. 첫 40문항의 2×2에서 nonzero N/J 또는 action-preference 변화가 극히 드물면 원인 audit을 먼저 한다. 고정 후보까지 확대해도 독립 source/질문 신호 사례가 충분하지 않으면 stage2 학습을 강행하지 않고 `not_run: stage1_gate_failed`로 기록한다. 작은 pilot 운영상 독립 질문 10개 미만의 신호로는 head 비교를 가설 검증으로 진행하지 않는 것을 권고한다. 충분한 학습의 최종 기준은 확대 train에서 독립 신호 사례 약 30개 확보 여부와 source 분포를 함께 보며, dev의 positive 개수에 맞춰 표본을 선택하지 않는다. 표본 확대 여부와 기준을 첫 결과 전에 manifest에 고정한다.
3. baseline 대비 개선 방향이 불명확하면 작은 dev의 좋은 seed만 택하지 않는다. 미리 정한 소수 seed를 동일하게 실행하고 모든 결과를 남긴다. 2pp 비열등성이나 20% 지연 절감의 검증은 이 규모로 완료됐다고 하지 않는다.

한 boundary에서 2개 action의 2×2는 query 4회 + answer 8회 + stop answer 2회, 즉 **문항당 최대 14 VLM 호출**이다. 50문항이면 자연 rollout과 oracle 진단을 제외하고 최대 700회다. 동일 D/입력은 재사용 가능하나 실제 호출 절감량을 기록한다. calibration에서 잰 평균 호출 시간으로 총 예산을 다시 계산한다. 2주 계획의 200–300문항, 다수 boundary, 모든 baseline을 한 대화 턴에 끝낼 수 있다고 미리 약속하지 않는다. 반대로 사용자가 이미 승인한 작은 1·2단계는 도구·시간이 허용하면 멈추어 재승인을 요구할 이유가 없다.

## 6. 완료로 인정할 기록

최소 산출물은 `manifest`, 원시 `reader_oracle` 결과, `access_2x2` 원시 cells/query/result/input hash, 질문 단위 split, 모든 baseline head 설정과 checkpoint, dev closed-loop 예측, QA–wall 비교, 데이터/모델 복구 검증 및 gate 판정이다. README의 완료 표현은 실제 실행 범위와 맞춘다. gate 실패를 발견해 근거와 수정 후 결과를 기록한 것도 실행 검토의 정당한 결론이다. `학습 완료`와 `가설 지지`는 별개의 상태로 관리한다.
