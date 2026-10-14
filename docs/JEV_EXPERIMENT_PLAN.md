# jev_v1.0 실험 계획 — 로컬 Gemma Reader의 decision head

2026-10-14 · 브랜치 `jev_v1.0` · 상태: **계획·기존 로그 감사 완료, 새 모델 학습·속도 비교는 아직 실행하지 않음**.

목표는 기존 A-RAG의 멀티모달 입력을 유지하면서 Gemma Reader가 중간 턴의 행동을 직접 선택하도록 만드는 것이다. 먼저 구현 가능성과 전체 지연을 검증하고, 이후 절약한 계산을 근거 탐색·검증에 사용해 기존 실패 문항을 해결할 수 있는지 시험한다.

## 1. 고정 범위

| 구성 | 결정 |
|---|---|
| Reader·학습 대상 | **로컬 `google/gemma-4-E4B-it`**, 기존 고정 revision 유지 |
| 1차 학습 | Reader backbone 동결 + 최종 표현의 작은 decision head |
| 후속 학습 후보 | 라벨 확보 후 Gemma의 제한된 LoRA + head; 전체 미세조정·RL은 초기 실험에 포함하지 않음 |
| 검색 | 기존 G1: EmbeddingGemma 2 텍스트+실제 이미지, 기존 인덱스·RRF 유지 |
| 실행 | 같은 Gemma가 선택도 하고 필요시 툴 인자·최종 답변도 생성 |
| 외부 모델 | **OpenAI·Jev 등의 외부 API는 실험에 사용하지 않음**. 문서·코드는 설계 참고용 |
| 데이터 | M3DocVQA, 같은 288문서 코퍼스. 질문은 텍스트, 검색한 실제 이미지는 Reader 입력 |
| 반복·근거 | A-RAG max iteration 8 + 별도 강제 최종 호출, 이미지 최대 8개, soft tokens 1120 유지 |

기존 임베더 비교 결과는 main에서 유지하고, 이 브랜치의 decision 실험 결과와 섞지 않는다.

## 2. Astra 검토와 기존 궤적 감사

Astra(`gpt-6-astra`)가 로컬 코드·로그를 읽고 설계를 검토했다. 판정은 **audit·작은 head의 실행 가능성 검증은 Go, 현재 라벨만으로 일반화·calibration·속도 개선을 주장하는 것은 No-go**다.

| 이전 검색 조건 | M3 문항 | 성공 궤적 | Reader 호출 |
|---|---:|---:|---:|
| B0 | 100 | 2 | 339 |
| C1 | 100 | 0 | 334 |
| G1 | 100 | 7 | 282 |
| 합계 | 같은 100문항 × 3조건 | **9개** | **955** |

955회에서 관측된 행동은 `keyword_search`, `semantic_search`, `read_chunk`, `FINAL`뿐이며, 한 턴의 복수 툴 호출은 0이었다. 따라서 8개 툴 조합을 학습한 것처럼 다루지 않는다. 원래 Agent의 복수 호출 기능은 유지하고, head가 지원하지 않는 상황에서는 기존 생성 경로로 돌아간다.

### 결과와 무관하게 먼저 고정한 분할

[split.json](../lab/registry/jev_v1.0/split.json): qid 정렬 → seed 20261014로 섞기 → train 60 / development 20 / test 20. 같은 qid의 B0·C1·G1은 같은 분할에 둔다. 분할 후 train의 성공 궤적만 학습 라벨로 수확했다.

| 분할 | 성공 궤적 / 서로 다른 성공 문항 | keyword | semantic | read | FINAL |
|---|---|---:|---:|---:|---:|
| Train | **5 / 4** | 5 | 5 | **0** | 5 |
| Development | 3 / 3 | 2 | 4 | 2 | 3 |
| Test | 1 / 1 | 1 | 2 | 1 | 1 |

[감사 원자료](../lab/reports/jev_v1.0/teacher_audit.json). **현재 train은 15턴뿐이고 read_chunk 양성 라벨도 없다.** 이 상태에서 4-class head의 유효성이나 안정적인 확신도 임계값을 주장하지 않는다. 분할 seed를 바꾸거나 dev/test 성공을 train으로 옮겨 부족한 클래스를 채우지 않는다.

이 100문항은 이전 실험에서 결과를 이미 관찰한 개발 자료다. 새 연구 안에서는 train/test를 구분하지만 완전히 새로운 비공개 테스트라고 부르지 않는다. 같은 코퍼스의 문서 공유는 별도 집계하고, 문항 holdout과 새로운 문서 일반화도 구분한다.

## 3. 데이터·라벨 확보 순서

1. 15턴으로 head 학습·저장·로드의 동작만 검증한다. 미관측 read 클래스 때문에 일반 정책 비교는 통과시키지 않는다.
2. 기존 100문항과 겹치지 않고 준비된 코퍼스에서 근거를 찾을 수 있는 M3 학습 후보를 결과를 보기 전에 고정한다. 최대 100개를 1차 수집 예산으로 두고, 후보 수·문서 범위를 먼저 확인한다.
3. **학습 후보에서만** 원래 Gemma A-RAG를 문항당 한 번 실행해 성공 궤적을 수확한다. 오답을 재시도해 맞춘 버전을 고르지 않는다. 근거·정답 metadata는 teacher의 실제 입력에서 제외한다.
4. 본 4-class 비교 진입 조건은 **서로 다른 성공 학습 문항 30개 이상, 각 클래스가 10개 이상의 서로 다른 성공 문항에서 관측**되는 것이다. 이는 사전 설정한 PoC 최소 조건이며 통계적 충분성을 보증하지 않는다. 수집 예산 내 충족하지 못하면 부족 자체를 결과로 보고한다.
5. 실패 궤적의 행동을 섞는 전체 teacher imitation은 별도 대조군으로만 둔다. 성공 라벨과 같은 의미의 정답으로 합치지 않는다.

라벨은 ‘성공한 teacher가 그 상태에서 선택한 행동’이다. 최적 행동이라는 뜻은 아니다. 성공 궤적의 불필요한 검색이나 잘못된 중간 행동까지 모방할 위험이 있다. 후속 탐색·기억 제어의 정답 라벨은 단순 툴 이름 라벨에서 자동으로 생기지 않는다.

## 4. 주 구현

### 4.1. 첫 head: 고정된 네 행동

기존 Reader의 **정확한 generation boundary**에서 최종 정규화 hidden vector를 읽는다. 같은 전체 멀티모달 forward의 마지막 유효 토큰 표현만 사용한다.

$$
\mathbf{z}=W\mathbf{h}+\mathbf{b},
\qquad
p(a\mid s)=\operatorname{softmax}(\mathbf{z})_a
$$

$$
\mathcal{L}_{\mathrm{CE}}=-\log p(a_{\mathrm{teacher}}\mid s)
$$

네 행동은 keyword / semantic / read / FINAL이다. 먼저 정규화된 linear head와 고정 학습 seed를 사용하고, epoch·정규화 강도는 development에서만 선택한다. backbone·vision 인코더는 동결한다. 분류 성능을 확인하기 전에 LoRA나 RL 복잡도를 추가하지 않는다.

이 확률은 **teacher 행동 선택 분포**다. ‘답변이 정답일 확률’이나 ‘근거가 충분할 확률’과 같지 않다. 종료 판단·근거 충족도 head는 별도 라벨·평가가 필요하다.

### 4.2. 선택 뒤 툴 인자는 여전히 필요하다

- keyword에는 keywords, semantic에는 query, read에는 실제 검색 결과의 chunk_ids가 필요하다.
- 주 경로는 같은 prefill에서 얻은 KV를 재사용해 선택한 툴의 **native tool-call prefix** 뒤 인자를 생성한다. FINAL이면 같은 모델이 최종 답변을 생성한다.
- 강제로 넣은 prefix 토큰 처리·인자 생성·검증의 시간과 토큰도 모두 비용에 넣는다.
- KV 분기나 Gemma 전용 캐시 처리가 검증되지 않으면 먼저 이중 prefill을 사용하는 보수적 구현으로 동작을 확인하고 그 추가 비용을 그대로 기록한다.
- teacher의 미래 검색어·인자·정답 이미지·최종 답을 현재 상태의 후보로 주지 않는다.

이미 정해진 구체적인 행동 후보를 선택하면 인자 생성도 없앨 수 있지만, 후보 생성 정책이 추가되어 문제 자체가 달라진다. 이 방법은 나중에 **별도 제한 행동 공간 실험**으로 비교한다.

### 4.3. Gemma 구현에서 주의할 사항

전체 Gemma4Model 경로를 유지해 vision·PLE를 우회하지 않는다. `output_hidden_states=True`로 42층의 전체 시퀀스를 저장하지 않고 마지막 norm hook 또는 모델 출력의 마지막 토큰만 읽는다. 필요시 LM logits는 `logits_to_keep=1`로 제한한다.

first turn / tool response / 이미지 추가 / 강제 FINAL의 실제 processor template을 검증한다. `<|turn>model`을 임의로 이어 붙이지 않는다. 기본 이미지 soft-token 280과 기존 1120을 혼동하지 않는다.

기존 history와 이미지·tool definitions는 유지한다. 단일 토큰 선택 안내를 붙이는 비교군은 입력 토큰이 달라지므로 ‘완전히 동일한 입력’이라고 표기하지 않는다.

## 5. 비교 조건

| 조건 | 역할 | 허용되는 해석 |
|---|---|---|
| B-HF | 같은 Gemma·HF backend의 원래 native 툴 호출 생성 | 주 속도·품질 기준선 |
| B-vLLM | 기존 실행 방식 | 시스템 참고치. HF 대비 차이를 head 효과로 해석하지 않음 |
| D-head | 동결 Gemma + 작은 행동 head + 인자·최종 생성 | 같은 backend 기준 head 효과 |
| D-logit | 학습 없이 기존 출력 logits를 읽는 제약 선택 | 추가 prompt·prefix 평가 비용과 조건부 확률을 명시한 진단 |
| D-LoRA | 충분한 라벨 확보 후 Gemma LoRA + head | 후속 조건; 초기 결과와 섞지 않음 |

초기 주 비교는 B-HF ↔ D-head다. D-logit은 참고 진단이며 ‘외부 decision 모델을 붙인 결과’가 아니다. 기본 head와 학습되지 않은 class를 숨겨 임의 정확도를 만들지 않는다.

## 6. 단계별 실행·중단 기준

| 단계 | 수행 | 다음 단계 진입 조건 |
|---|---|---|
| A · CPU 감사 | 분할·라벨·기존 생성 내용/툴 인자 토큰 audit | 누수 없음, 부족한 class 공개 |
| B · 입력 검증 | template, 실제 이미지, native tool parser, 강제 FINAL 테스트 | 기존 입력·툴 의미 보존 |
| C · GPU smoke | text/image 상태의 HF forward·head·KV 이어쓰기 | 오류 없이 동작, VRAM·시간 실측 |
| D · 작은 학습 | train 성공 라벨의 head 학습·미관측 클래스 진단 | 4-class 일반 평가 전 라벨 진입 조건 충족 |
| E · 오프라인 평가 | teacher-state action agreement·혼동·종료 오류 | threshold·하이퍼파라미터를 test 전에 고정 |
| F · 폐루프 비교 | 고정 test 20문항, 조건별 문항당 한 번 | 실패도 제외하지 않고 모두 결과에 포함 |
| G · 기여 검증 | 같은 시간 예산의 추가 탐색·기억 제어 | 속도 PoC와 별도의 분할·라벨·평가 설계 |

GPU는 공유 자원이다. 다른 세션의 프로세스를 종료하지 않는다. 초기 우리 프로세스의 목표 상주 사용량은 25GB 이하이며 실제 peak allocated/reserved와 전체 GPU 여유를 확인한다. 메모리 여유 부족·OOM 가능성·지속적인 자원 경합은 사용자에게 바로 알리고 새 GPU 단계를 시작하지 않는다. CPU 감사·문서·입력 검증은 계속할 수 있다.

## 7. 평가 지표와 공정한 시간 측정

### 답변과 행동 품질

EM·F1, 이미지 전달 recall(완전한 이미지 대응이 있는 문항 수 별도), 툴 인자 유효성, 검색 실패, 반복 행동, 최대 iteration 도달, fallback 비율을 기록한다. 오프라인 teacher 행동 일치율과 폐루프 정답률을 분리한다.

premature FINAL은 단순히 ‘teacher보다 일찍 멈췄다’와 ‘일찍 멈춰 오답이 됐다’를 구별한다. head 확률의 calibration은 먼저 teacher label 대상 NLL/Brier·reliability plot을 본다. 정답 확률의 calibration이라고 부르지 않는다. 현재 dev의 성공 라벨 수로 안정적인 temperature·임계값을 맞췄다고 주장하지 않는다.

### 전체 지연

$$
\begin{aligned}
T_{\mathrm{total}}&=T_{\mathrm{prefill}}+T_{\mathrm{decision}}+T_{\mathrm{arguments}}\\
&\quad+T_{\mathrm{tools}}+T_{\mathrm{final}}+T_{\mathrm{fallback}}
\end{aligned}
$$

이 항들은 겹치지 않는 실제 계측 구간으로 정의한다. 공통 prefill을 decision과 인자 양쪽에 중복 집계하지 않는다. 실제 이중 prefill은 두 번 모두 포함한다. 모델 로드·오프라인 학습·인덱스 준비 비용은 별도 표에 둔다.

같은 backend·GPU·warm-up·이미지 입력·출력 한도·cache 정책으로 비교하고, 문항별 조건 실행 순서는 고정 seed로 교차 배치한다. 동시에 여러 조건을 돌려 서로의 지연을 늘리지 않는다. 전체 wall time, 모델 호출 수, prefill/decode token, 이미지 인코딩, warm/cold, 다른 GPU 프로세스의 존재를 기록한다.

20문항에서는 개별 paired 결과와 중앙값·질문 단위 bootstrap 구간을 중심으로 보고하고 p95·비열등성·배포 가능성을 강하게 주장하지 않는다. ‘decision이 빠르다’가 아니라 **인자와 fallback까지 포함한 실제 풀이가 빨라졌는지**를 판정한다.

## 8. 속도를 넘어서는 후속 가설

우선순위는 [contribution 문서](JEV_CONTRIBUTIONS.md)를 따른다. 현재 Reader가 근거를 받았는데도 틀리는 문제에 대해, 같은 시간 예산 안에서 근거 충족·상충 검사를 더 자주 하거나 탐색을 재배치하는 것을 시험한다. 저장·수정은 실제 장기 기억 전이 데이터가 필요하며 고정 PDF 검색 점수만으로 검증하지 않는다.

## 9. 산출물·현재 상태

- 완료: 브랜치, Astra 계획 검토, 기존 궤적 감사, outcome-independent split, 최신 decision 학습법·RAG/memory 조사 문서.
- 미완료: 새 head 구현·학습·폐루프 속도/정확도 비교. 문서의 가설을 성능 결과로 쓰지 않는다.
- 실행 전에 코드·입력·모델 revision을 고정하고 새 output identity를 사용한다. 원래 보고서·완료 예측은 수정하지 않는다.

참조: [decision 모델 학습법](JEV_DECISION_MODEL_TRAINING_REVIEW.md) · [Astra 최신 연구 메모](JEV_MEMORY_RESEARCH_ASTRA.md) · [기존 A-RAG 코드](../src/lsdrag/agent.py).
