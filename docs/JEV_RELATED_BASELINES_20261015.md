# Full-tool A-RAG와 멀티모달 JEV: 비교군·데이터 선정 검토

기준일 2026-10-15 UTC. 원 저자 논문·코드·dataset card를 확인한 연구 설계 문서다. 폐기된 3-action 실험의 수치·결론을 재사용하지 않는다. 새 full-tool 실험을 실행하거나 결과를 주장하지 않는다. 문서의 권고는 선행연구와 자원 조건을 바탕으로 한 판단이다.

## 권고하는 최소 범위

**최소 두 환경군: 멀티모달 RAG 하나와 비RAG의 로컬 stateful tool agent 하나.** 두 환경에서 생성형/한 토큰/BC/value/제안 supervision을 비교한다. 주장은 tool calling 구현이나 지연 감소 자체가 아니라, 관측에 따른 다음 행동의 가치 구분을 학습해 더 나은 선택을 하는 원리여야 한다. RAG 권고는 MC-Search feasibility를 먼저 확인하고, 비RAG 권고는 BFCL V3 multi-turn의 로컬 API subset이다. Memory와 여러 browser/robotics 환경을 동시에 추가하지 않는다. 아래 QA 데이터 표는 후보군이며 모두 실행할 목록이 아니다.

주 구현은 vanilla A-RAG의 keyword/semantic/read 구조에 G1의 이미지 fusion·자동 전달을 유지하고, 명시 image_search/figure_read를 추가한다. 이 변경은 **우리의 멀티모달 A-RAG adaptation**이다. 원 논문이 같은 다섯 도구를 제공한다고 표현하지 않는다. 공식 [A-RAG 논문](https://arxiv.org/abs/2602.03442)과 [저장소](https://github.com/Ayanami0730/arag)는 세 도구와 다중 granularity의 자율 탐색을 제시한다. 원 코드의 `chunk_read`와 로컬 `read_chunk` 명칭 차이는 adapter 표에 명시한다. 원 A-RAG 프로토콜 재현과 공통 도구 프로토콜의 비교는 서로 다른 실험이다.

## 반드시 남길 core matrix

모든 행은 같은 질문 split, frozen reader/retriever, tool registry·prompt contract, legal action mask, query/argument generation, 이미지 예산, 종료 예산, 최종 답변 형식, cache 정책을 사용한다. 학습 controller가 더 좋은 retrieval kernel을 받으면 controller 효과로 해석할 수 없다.

| 비교군 | 결정 방법 | 직접 확인할 주장 |
|---|---|---|
| Native/generated A-RAG adaptation | 모델의 실제 native tool call 또는 공통 생성형 controller | 실제 사용자가 얻는 QA–latency 기준선. parser·fallback과 tool-name/인자 token 비용 포함 |
| Constrained first-token | 유효한 각 행동을 단일 token option으로 읽고 같은 인자 경로 실행 | 긴 행동 설명을 줄이는 것만으로 충분한지. option 순서·contextual tokenization 검증 |
| BC linear/작은 MLP | frozen state에서 behavior action 분류 | 복잡한 목표보다 간단한 imitation이 충분한지. 성공 궤적만이 아니라 전체 행동 데이터 사용 |
| Generic return/Q head | 같은 상태와 lawful factual branch return·비용으로 학습 | 가장 중요한 알고리즘 기준선. 현재 evidence를 읽는 작은 head라는 구조만의 효과 분리 |
| Generic Q + paired states | S0/S1 factual returns를 모두 제공 | 제안의 이득이 evidence-removal augmentation 때문인지 |
| Proposed access auxiliary | 같은 Q backbone에 module-access 2×2 대조 보조 supervision | off-diagonal label이 새 문항의 실제 정책에 유용한지 |
| Fixed-budget/query-only policies | 항상 멈춤, 정해진 호출 수, 초기 query만 이용한 선택 | 상태 적응이 정말 필요한지; 비용–품질 reference curve |

같은 full-tool action set에서 `FINAL`과 다섯 도구를 학습·배포한다. 검색 이후 `read_chunk`/`figure_read` 선택과 검색 tool 선택을 섞어 3-action 대리 task로 바꾸지 않는다. 최적 문서 ID/새 query를 head가 자동으로 해결한다고 가정하지 않는다. 후보 pointer를 추가하면 인자 생성 제거를 따로 ablate한다. **후보 shortlist와 pointer 비용을 모든 비교군에 공정하게 맞추기 전에는 다른 acquisition kernel**이다.

### 같은 원자료와 같은 수집 비용은 별도 비교

제안만 2×2 실행을 더 수행하면 더 많은 teacher 비용을 쓴다. (1) 모든 learned baseline이 같은 factual states/returns를 받고, (2) access-aux 수집에 쓴 GPU-hour로 generic Q가 자연 branch를 더 받는 equal-budget 비교를 각각 둔다. BC는 같은 실패·복구 상태를 보되 제안의 utility target을 action ground truth라고 이름만 바꾸지 않는다. 선택된 seed·λ를 test에서 다시 고르지 않는다. action probabilities, Q preference, 정답 확률을 구분한다.

## 외부 방법 비교: 우선 하나만

[Stop-RAG 원문](https://arxiv.org/html/2510.14337v1)과 [공식 코드](https://github.com/chosolbee/Stop-RAG)는 STOP/CONT의 미래 return을 full-width Q(λ)로 학습한다. 따라서 “현재 confidence 대신 미래 utility로 정지” 자체는 새 기여가 아니다. full-tool 환경으로 옮기면 **Stop-RAG-inspired multi-action adaptation**으로 명명하고 원 논문의 숫자를 같은 실험 결과처럼 인용하지 않는다. 원 저장소는 4 H100으로 실행했다고 명시하므로 한 GPU에서 원 전체 pipeline·원 학습 규모 재현을 필수로 약속하지 않는다. 작은 공통 frozen feature/Q head로 target 원리를 비교하는 것이 우선이다.

[MC-Search](https://openreview.net/forum?id=JEGDp1E4OH)는 cross-modal 순차 탐색과 structured chain supervision을 다룬다. [공식 프로젝트](https://mc-search-project.github.io/)와 [코드](https://github.com/YennNing/MC-Search)가 있어 외부 멀티모달 task/agent reference로 적합하다. 그러나 Search-Align 전체 학습까지 동시에 재현하는 대신, 데이터와 평가·agent pipeline을 먼저 연결하고 핵심 제어 baseline에 집중한다. navigation 역할·증거 제거와의 중복은 기존 연구 제안/Astra 검토에 이어 별도 방법 novelty 감사가 필요하다.

CoT를 주제에 추가할 필요는 없다. 원 질문/관측으로 query를 만드는 경로와 controller 행동 선택을 분리하면, 생성형 baseline의 reasoning budget을 작게/크게 바꾸는 sensitivity로 처리할 수 있다. 생각 토큰을 많이 쓰는 baseline 하나만 고르고 head가 빠르다고 주장하지 않는다.

## 데이터 후보와 실제 접근성

“gold support/중간 답”과 “최적 tool trajectory”는 다르다. 아래 QA benchmark의 증거 표시는 controller의 최적 행동 label이 아니다. gold로 고른 branch만 학습하거나 gold chain을 runtime prompt에 넣지 않는다.

| 데이터 | 실제 제공물·접근 | supervision·라이선스 | 권고와 비용 |
|---|---|---|---|
| **MuSiQue** | [공식 GitHub](https://github.com/StonyBrookNLP/musique)의 데이터 download 및 evaluator | multi-hop 질문·답·paragraph와 decomposition/support 정보. repository license CC-BY-4.0. decomposition은 최적 도구 실행 이력 아님 | 텍스트 외부 검증 1순위. 문항 합성 기반 및 source 공유를 고려한 분할. 전체 retriever 구축보다 사전 고정 shared corpus pilot부터 가능 |
| **HotpotQA** | [공식 홈페이지](https://hotpotqa.github.io/)가 train/dev와 fullwiki corpus 공개 | 답과 sentence-level supporting facts; CC-BY-SA-4.0 명시. distractor/fullwiki 설정 구분 필수 | 널리 알려진 second text reference. fullwiki 구축은 storage/index 비용이 큼. 작은 distractor 설정을 fullwiki 결과로 표시하지 않음 |
| **2WikiMultihopQA** | [공식 GitHub](https://github.com/Alab-NII/2wikimultihop)의 data.zip/data_ids와 evaluation v1.1, hyperlink corpus links | answers·supporting_facts·evidence triples·일부 entity aliases. repository Apache-2.0; 원 Wikipedia/Wikidata 권리 별도 보존 | 관계 추론 실패와 검색 제어 분리에 유용. triples/aliases를 controller input으로 전달하면 누출. MuSiQue와 동시 주력보다 후속 검증 |
| **A-RAG reformatted suite** | [공식 HF](https://huggingface.co/datasets/Ayanami0730/rag_test) `chunks.json`, `questions.json`; 텍스트 세 세트 각각 1,000Q의 축소 corpus | card Apache-2.0, 원 데이터 출처 별도. 예시의 evidence field는 비어 있으므로 gold supports/trajectories 존재를 가정하지 않음 | 구현 smoke·원 A-RAG 비교에는 편리. 이미 평가용으로 재구성된 질문을 train/test로 쪼개고 “공식 train/test”라고 부르지 않음 |
| **MultiModalQA** | [공식 GitHub](https://github.com/allenai/multimodalqa) QA/text/table/image metadata와 실제 image ZIP 공개 | final answers·supporting_context·metadata.intermediate_answers. GitHub 포괄 dataset LICENSE 없음; Wikipedia/Wikimedia 원출처별 attribution 보존 | controlled feasibility에 적합. **metadata.table_id의 공개 표는 실제 baseline 입력**이므로 반드시 공통 배경 B에 유지. metadata.modalities만으로 표가 불필요하다고 판단하지 않음. candidate IDs와 중간 답은 private audit로 분리 |
| **M3DocVQA** | [공식 코드](https://github.com/konnisan/m3docvqa), [공식 연결 HF assets](https://huggingface.co/datasets/ad6398/m3docvqa)의 MMQA files·PDFs·dev_doc_ids | MMQA 질문을 PDF corpus로 확장; 코드 및 HF card Apache-2.0지만 원 페이지 내용 권리는 별도. 정답 근거는 tool trajectory 아님 | 기존 pipeline 자산과 잘 맞음. page rendering/visual indexing가 주비용. MMQA와 **같은 질문 계보**이므로 독립 데이터 일반화 증거로 두 개를 세면 안 됨 |
| **MC-Search** | [공식 HF](https://huggingface.co/datasets/YennNing/MC-Search)에 8 parquet shards, KB all_docs/all_image_infos 및 embeddings archive 실제 listing 확인 | card `subqa_chain`에 modality/subquestion/intermediate answer 포함. KB 14,037 passages/10,953 image metadata. HF card license 미명시 | full-tool 멀티모달 새 main task의 우선 feasibility 후보. 공개 listing이 실제 모든 corpus image bytes 재현을 보장하지 않으므로 URL 만료·재다운로드·license·gold image input 경계 먼저 audit. chain은 reference reasoning, 우리 kernel의 최적 행동 경로 아님 |
| **Mem-Gallery** | [공식 HF](https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery)의 dialog/image와 [평가 코드](https://github.com/YuanchenBei/Mem-Gallery) 공개 | card MIT. 20 scenario, multi-session conversations·images·QA. 최적 read/write tool trajectory benchmark라고 주장하지 않음 | memory 역할을 주기여로 바꾸지 않는 한 optional transfer만. 같은 persona/scenario의 여러 QA를 독립 수천 표본으로 세지 않음. 원 이미지와 caption의 visual necessity audit 필요 |
| **MADQA, 2026** | [공식 코드](https://github.com/OxRML/MADQA), [논문](https://arxiv.org/abs/2603.12180), [HF dataset](https://huggingface.co/datasets/OxRML/MADQA) | 2,250 human-authored Q/800 PDF. 코드 Apache-2.0지만 dataset card **CC-BY-NC-4.0**이고 README asset 요청은 인증/접근 제한 응답 확인. Page F1 등 평가 제공 | 최신 agentic document QA 외부 검증 후보. 접근 확인과 corpus 다운로드 전 주 일정에 의존하지 않음. 관측된 사람/agent trace가 있어도 optimal policy ground truth와 다름 |

Dataset card의 라이선스와 코드 라이선스를 혼동하지 않는다. 위 “미명시”는 사용이 금지됐다는 판정이 아니라 검증하지 않은 재배포 권리를 주장하지 않겠다는 뜻이다. 이번 확인은 listing·문서 검토이며 새 전체 corpus를 내려받아 byte completeness를 검사한 것은 아니다.

### 필수 두 번째 환경: 비RAG stateful tool agent

**BFCL V3 multi-turn의 로컬 실행 API를 우선 권고한다.** [공식 V3 설명](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)은 file system·vehicle·trading·travel 등 시뮬레이션 API를 여러 단계로 호출하고, 실제 종료 API state와 실행 이력을 평가한다. 단일 호출 AST benchmark와 다르다. [공식 구현](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)은 공개 data·실행 클래스·평가기와 `bfcl-eval` 설치 경로를 제공한다. V4의 web-search/memory 대신 **고정 V3 multi-turn local categories만** 선택하면 SerpAPI·외부 사이트·음성·사용자 simulator LLM을 필수 의존성으로 두지 않아도 된다. 정답 예제 trajectory는 reference 실행 경로이며 유일한 최적 정책이라고 가정하지 않는다.

이는 새로운 BFCL leaderboard 점수를 내기 위한 선택이 아니다. 원 V3 사례는 공개 evaluation data이므로 이를 임의로 학습/평가에 나누면 **BFCL-derived adaptation**이다. 권고는 공개 API engine으로 별도의 training initial states/tasks를 생성하고, 평가 task templates·initial states·domain을 사전 분리하는 것이다. 생성 task의 성공 조건은 executable checker로 검증하며, 원 reference와 겹치는 template·parameter 조합을 기록한다. 이 작업이 불가능하면 환경의 declared resplit 범위로 claim을 좁히고 공식 test 일반화라고 부르지 않는다.

| 두 환경의 공통 원리 | RAG에서 구현 | 비RAG에서 구현 |
|---|---|---|
| 새 관측이 다음 행동의 가치를 어떻게 바꾸는가 | 검색/이미지 evidence를 얻은 뒤 다음 search/read/FINAL advantage | API result를 얻은 뒤 다음 inspect/mutate/finish advantage |
| utility | 공식 QA score와 이후 실행 비용 | executable task success·state correctness와 이후 실행 비용 |
| branching | 같은 prefix, 같은 corpus/예산에서 실제 도구 실행 | engine state를 복제/복구한 동일 prefix에서 실제 로컬 API 실행 |
| 일반화 | 새 질문·문서/source·가능하면 새 corpus | 새 task templates/initial state; 후속 held-out domain/tool set |
| 배포 | 관측 상태와 도구 schema를 읽어 action 선택, 필요한 인자만 생성 | 같은 원리의 schema-conditioned action 선택과 인자 생성 |

**중요한 설계 미해결점:** 기존 RAG의 query-generator×answer-reader 접근 2×2를 stateful task에 그대로 옮길 수 없다. API 최종 state checker는 LLM answer reader가 아니며, 마지막 답변에서 관측을 숨겨도 이미 실행된 state change가 사라지지 않는다. 따라서 해당 두 모듈 이름을 바꾸는 것으로 일반 원리가 입증됐다고 하지 않는다. 공통 연구 질문을 “새 관측이 이후 행동 advantage에 주는 효과”로 정의하고, 비RAG에서는 planner/argument generator의 observation access를 개입하되 실제 state reward를 측정한다. RAG off-diagonal auxiliary가 이 공통 신호에 추가 가치를 주는지 별도로 검사한다. **동일한 supervision 원리가 비RAG로 타당하게 정의되지 않으면 현재 방법은 RAG-specific이며 broad-agent claim을 철회하거나 방법을 다시 설계해야 한다.** 이는 실행 전에 해결할 이론·프로토콜 검토 사항이다.

새 tool set까지 일반화하려면 고정 이름별 linear output만으로 부족하다. schema/description 기반 shared scorer 또는 dynamic candidate readout을 generic Q와 제안 모두에 동일 제공한다. 우선 환경별 head 재학습으로 같은 학습 원리가 재현되는지를 검증하고, RAG-trained weights의 zero-shot API 성공과 구분한다. 이름만 바꾼 tool permutation, 같은 도메인의 새 함수, 완전히 새 domain을 단계별로 분리한다.

다른 환경은 다음 우선순위다. [ALFWorld](https://github.com/alfworld/alfworld)의 text-only engine은 로컬 CPU 환경·공개 task/data·agent baseline을 갖추어 외부 API 없이도 embodied-style 행동 제어를 시험할 후보이며, 시각적 THOR backend를 요구하면 별도 자원이 필요하다. [ScienceWorld](https://github.com/allenai/ScienceWorld)는 text-world science tasks를 제공하고 Java 환경 준비가 필요하다. [BrowserGym](https://github.com/ServiceNow/BrowserGym)은 브라우저 task 공통 인터페이스지만 WebArena/OSWorld 계열은 사이트/VM/화면·브라우저 세팅과 재현 비용이 추가된다. 단일 GPU 문제보다 환경 provisioning이 병목일 수 있다. [τ-bench 공식 구현](https://github.com/sierra-research/tau2-bench)은 사용자/도구 상호작용과 정책 제약이 더 풍부하지만 사용자 simulator·동시 모델 호출·버전별 grading을 통제해야 한다. 현재는 이들을 전부 추가하지 않는다.

[ToolBench](https://github.com/OpenBMB/ToolBench)는 API tool-learning과 trajectory의 관련 연구지만, live API 변화·실패·remote 비용 때문에 첫 비RAG 환경으로는 BFCL local engine보다 불리하다. Gorilla/ToolBench code의 Apache-2.0과 개별 외부 API 권리는 구분한다.

### “더 나은 선택”과 “왜”를 별도로 입증

QA/task success가 유지되면서 latency가 줄어드는 것은 효율 결과다. 결정 품질 향상은 다음과 같이 따로 검사한다.

1. 동일 frozen prefix에서 lawful alternative action을 실제 실행해 cost-adjusted return과 비교한다. chosen-action return, bad continuation rate, premature finish rate, observed branch regret를 보고한다. noisy single-run best를 기대 최적 정책으로 부르지 않는다.
2. 관측을 제거/유사 크기 distractor로 치환할 때 다음 행동과 return이 어떻게 변하는지 본다. 효과 부호·크기 예측, 행동 preference 변화, 성공/실패 사례를 실패 포함 자연 분포에서 분석한다. 간단한 관련성/현재 confidence로 설명되는지도 대조한다.
3. learned return 또는 auxiliary 예측만 좋아지고 실제 closed-loop 행동·task success가 좋아지지 않으면 실패다. 선택을 강제로 맞춘 replay로 인자 생성·답변 형식 변화와 controller 효과를 구분한다.
4. 접근 대조의 teacher GPU-hour, paired-state augmentation, 임의 auxiliary target과 동일 architecture를 맞춘다. “왜”는 모델이 생성한 설명문이나 attention heatmap 하나가 아니라 개입 결과·예측된 action advantage·실제 실행 결과의 일치로 뒷받침한다.
5. 두 환경군에서 같은 원리의 β=0/β>0 비교가 필요하다. 비RAG에서 새 encoder·다른 reward shaping·더 많은 labels까지 제안에만 추가하면 transfer 근거가 아니다.

## 한 대의 96GB Blackwell에서의 모델 선택

[공식 `google/gemma-4-E4B-it` card](https://huggingface.co/google/gemma-4-E4B-it)를 확인했다. 모델은 실제 존재하며 native function calling, multimodal input과 thinking on/off를 지원한다고 명시한다. E4B는 effective parameter 이름이며 card의 embedding 포함 총량은 약 8B다. **E4B라는 이름만으로 4B weight 메모리를 계산하지 않는다.** 공식 model API에서 확인한 revision은 `ee0ef6023621cff504d758262d4e04895a5af4a2`, card license Apache-2.0이다.

[공식 `google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)의 확인 revision은 `914f7f89142e33e77833254d9c9b90c3cef7303b`다. 실제 text/image 인코딩 입출력과 fusion 설정은 현재 인덱스 fingerprint에 고정한다. text-only embedding을 multimodal embedding이라고 임의로 바꾸지 않는다.

권고는 **현재 검증 중인 Gemma backbone 하나를 고정**하는 것이다. 대형 모델 여러 개를 동시에 비교하는 대신 feature extraction·도구 호출·이미지 읽기까지 end-to-end가 제대로 작동하는지 확인한다. head 학습은 작은 CPU/GPU 작업이지만 branch teacher collection은 반복 VLM forward가 지배할 수 있다. 96GB는 VRAM 제약일 뿐 throughput·지연 보장이 아니다. 한 번에 reader와 embedder를 상주시킬지 순차 적재할지는 peak memory와 exclusive timing으로 결정한다.

두 번째 reader는 core 신호가 통과한 뒤 선택한다. 특정 미검증 새 모델 이름이나 양자화 설정을 지금 필수 baseline으로 추가하지 않는다. quantization·backend·thinking mode를 바꾸면 state representation과 학습 target이 달라질 수 있으므로 checkpoint 전용 head와 model identity를 다시 검증한다.

## 새 실험에서의 학습·평가 점검표

| 원칙 | 현재 확인할 것 | 별도 실험/다음 단계 |
|---|---|---|
| Reader floor | gold evidence **전체**와 benchmark-public context를 줬을 때 생성형 Reader가 풀 수 있는지; 원 이미지와 caption-only 차이 | Reader가 못 푸는 설정이면 head 학습 전에 task 또는 Reader 한 번 교체. controller로 perceptual failure를 해결했다고 주장하지 않음 |
| Registry 동기화 | schemas·prompt·head action order·runtime handlers의 단일 contract hash, G1 이미지 fusion/자동전달과 explicit tools의 실제 동작 | tool 추가/삭제 시 새 checkpoint/plan identity; 기존 4/3-action head를 잘못 연결하지 않음 |
| Split와 정보 경계 | question 전체 branches를 한 split에, source/entity/dependency 연결 성분 추적. public table, title, caption, query echo 포함 | shared corpus 검색으로 train feature에 dev support가 들어왔는지도 기록. qid 분리만으로 unseen-source 주장 금지 |
| Label | factual action returns와 off-diagonal access targets 분리; 성공/실패·무효 검색 포함 | 관측 best branch는 realized oracle이며 최적 기대 return이 아님. 추가 장기 rollout은 별도 budget |
| Head 학습 | train-only normalization, seed grid·λ fixed, same architecture/steps/data 대조, legal mask | grouped held-out return prediction과 source shift; pilot dev best seed를 새 test에 맞춰 재선택하지 않음 |
| 실제 정책 | 매 iteration full tools에서 fresh dispatch·argument/query·final answer 실행, closed loop | offline classification만 개선되면 controller 성공 아님. 첫 검색과 evidence-conditioned continuation 효과 구분 |
| 비용 | 실제 이미지 preprocessing, prefill, vocabulary projection/head, 인자 decode, 검색, answer, fallback/timeouts, GPU-hour | same backbone 최적화는 모든 정책 공통 적용. latency mean/p50/p95와 paired question differences, tail·실패 보존 |
| 통계·claim | 질문/source cluster가 독립 단위. 중간 state 수를 표본 수로 세지 않음 | 최종 비열등성 margin·표본 수·test split은 pilot과 별도 사전 고정; 작은 n에서 유의차 없음은 품질 동등성 근거가 아님 |

첫 의사결정은 task 이름의 유행성이 아니라 **원본 근거가 실행 가능하고, Reader가 해결할 여지가 있고, 다른 행동 선택이 return을 바꾸는 자연 상태가 충분한가**다. 이를 통과한 주 데이터 하나에서 generic Q와 first-token을 이기는지 확인한 뒤 범위를 확장한다.
