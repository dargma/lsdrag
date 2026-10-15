# Direct control 선행연구와 학습 체크리스트

기준일: **2026-10-15 UTC**. 1차 출처인 논문 원문·학회 페이지·저자 공식 저장소/model card를 확인했다. 논문 초록과 관련 방법 절 중심의 검토이며 모든 결과 재현을 뜻하지 않는다. 모델 카드의 성능은 저자 보고이고 독립 검증 결과가 아니다. 아래 전이·우선순위·단일 GPU 실행 가능성은 우리 코드와 자원에 대한 판단이다.

## 현재 방법의 정확한 경계

고정 기반은 A-RAG 계열 `keyword_search`, `semantic_search`, `read_chunk`에 `image_search`, `figure_read`를 추가한 **5 tools + FINAL**이다. G1 text/image fused semantic search와 search-linked image delivery를 유지한다. frozen multimodal backbone의 상태를 작은 head로 읽고 선택된 handler의 이름·wrapper를 Python이 구성한다. **선택된 도구의 인자 JSON과 최종 답변은 계속 생성**하며, 현재 안전한 구현은 결정과 인자 생성에 별도 prefill을 쓴다. candidate-ID pointer, 상태 cache, early exit, JEPA 학습은 현재 구현이 아니다. 기존 Q3 축소 실험 수치를 이 설정의 결과로 재사용하지 않는다. 접근효과 auxiliary도 아직 검증된 기여가 아니다.

Jev는 TypeSafe의 structured decision 모델 명칭이다. JEPA와 동일하다고 해석할 근거가 없다. **2610.03935의 최신 v2(2026-10-12)는 v1의 InnerJev 명칭을 SII-Decision으로 변경**했다. 과거 문서의 InnerJev를 인용할 때 버전을 함께 적는다. [최신 원문](https://arxiv.org/html/2610.03935v2)

## 가장 가까운 연구: 신규성 위협과 실행 가능한 비교

| 우선순위 | 확인한 제목·날짜·1차 출처 | 실제 메커니즘과 우리와의 차이 | 96GB Blackwell 한 장에서의 비교 판단 |
|---|---|---|---|
| P0 | **ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings**, 2023-05-19. [논문](https://arxiv.org/abs/2305.11554) | frozen LM에 tool token/embedding을 학습하고 선택 후 arguments를 생성한다. 우리의 action readout→args 생성 구조와 매우 가까워 '선택을 생성에서 분리'만으로 신규성 부족. 독립 Q head와 native tool-token head는 구현 차이일 뿐이다. | 같은 frozen Gemma에 6-way tool-token/readout 비교를 먼저 구성. 원 논문 전체 재학습보다 동일 데이터·backbone의 faithful adaptation을 명확히 표기. |
| P0 | **General Decision Models: Benchmarking and Insights Beyond Jev**, v1 2026-10-02, **v2 2026-10-12**. [§6 최신 원문](https://arxiv.org/html/2610.03935v2) | SII-Decision은 유효 option label의 첫 토큰 확률을 single prefill로 읽고 reasoning endpoint 분포를 증류한다. non-generation/readout/distillation 모두 이미 존재. 우리 과제는 반복 RAG의 선택과 그 결과의 학습이다. | 동일 Gemma first-token 6-way baseline 필수. 공개 4B decision model 외부 비교 가능; 27B는 추론만 별도 순차 실행 검토. 텍스트 모델에 이미지 입력을 몰래 생략한 비교는 부적절. |
| P0 | **LatentGate: Low-Latency Semantic Routing via Frozen-Backbone Probing of Small Language Models**, ACL Industry **2026-07**. [학회 원문](https://aclanthology.org/2026.acl-industry.153/) | frozen SLM mean pooling, PCA whitening, linear probe의 non-generative agent routing. frozen hidden+가벼운 head+빠른 routing은 이미 직접 제안됨. 반복 retrieval의 미래 효과 학습과 pixel evidence 제어는 다르다. | 같은 상태 데이터의 last-token linear / mean-pool linear / whitening+linear를 학습량 일치시켜 비교. 매우 저렴. train-only whitening 필수. |
| P0 | **Stop-RAG: Value-Based Retrieval Control for Iterative RAG**, **2025-10-16**. [논문](https://arxiv.org/html/2510.14337v1), [공식 코드](https://github.com/chosolbee/Stop-RAG) | 반복 RAG를 STOP/CONT MDP로 두고 full-width forward-view Q(λ) targets로 offline value를 학습. generic Q와 adaptive stop은 신규성이 아니다. 우리는 다섯 도구의 종류 선택과 멀티모달 acquisition까지 포함한다. | binary stop/continue를 공정한 축소 비교로 사용하고 계속 시 공통 planner 유지. 별도로 6-action generic outcome-Q를 같은 수집 rollout·head 크기로 비교. 원 논문의 반복 답변 샘플 비용도 공개. |
| P1 | **Luna-2: Scalable Single-Token Evaluation with Small Language Models**, **2026-02-20**. [논문](https://arxiv.org/abs/2602.18583) | shared decoder SLM과 lightweight LoRA/PEFT metric heads로 single-token 평가. tool-selection 품질 평가를 포함하나 환경 실행 정책과 동일하지 않다. | 평가 head와 action policy를 혼동하지 않기. 직접 baseline 우선순위 낮음; shared representation/adapter 비용 분석에 참고. |
| P1 | **Latent Action Reparameterization for Efficient Agent Inference**, **2026-05-18**, v2 방법 확인. [원문](https://arxiv.org/html/2605.18597v2) | 반복 trajectory를 compact latent action 단위로 표현해 horizon과 action tokens를 줄인다. 단일 현재 tool-name을 분류하는 우리 방식보다 행동 표현/다단계 압축 범위가 넓다. | tokenizer/모델/action interpreter 변경과 새 학습이 필요. 당장 모든 baseline에 넣기보다 향후 action compression을 주장할 경우 필수 대조. |
| P1 | **Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning**, **2025-06-10**. [논문](https://arxiv.org/abs/2506.09033), [공식 코드](https://github.com/ulab-uiuc/Router-R1) | LLM router가 think/route를 반복하며 여러 LLM을 선택·집계; outcome/format/cost reward. non-generative head가 아니고 호출 대상도 retrieval tool과 다르다. | 전체 RL+모델 pool 재현은 한 GPU 시간 부담 큼. 즉시 필수 비교는 같은 reader 생성 planner와 cost-aware Q. multi-model orchestration을 논문 범위에 넣을 때 재검토. |
| P1 | **SkillOrchestra: Learning to Route Agents via Skill Transfer**, **2026-02-23**. [논문](https://arxiv.org/abs/2602.19672) | execution experience에서 skill과 agent별 competence/cost를 모델링해 다중 agent를 선택한다. generic 저비용 routing 주장은 선점되어 있다. | skill inference/agent pool 추가는 현재 5-tool 문제를 바꾼다. 도구별 competence feature ablation의 동기 정도로 한정. |
| P1 | **When Should Multi-Round RAG Stop? Structured Stopping Judgments and Retrieval Reduction in Search-R1**, **2026-08-13**. [논문](https://arxiv.org/abs/2608.13237) | frozen Search-R1에 structured sufficiency/gap judge를 추가하고 trajectory 첫 STOP의 순차 평가를 강조한다. retrieval 감소가 전체 inference 비용 감소를 뜻하지 않는다고 명시한다. | small judge 가능하나 별도 judge prefill 비용 포함. 우리의 종료 평가는 state classification accuracy 대신 실제 rollout 첫 STOP 기준으로 해야 한다. |
| P2 | **Speculative Macro Commit for Faster Tool-Using Agents**, **2026-09-03**. [논문](https://arxiv.org/abs/2609.03236) | authoritative actor와 drafter가 isolated 환경에서 action chains를 예측·실행하고 조건부 commit. 직접 분류 head와 다른 runtime acceleration. | actor+drafter 동시 실행은 단일 GPU 경쟁을 유발. local retrieval의 tool latency가 작다면 이점 작을 수 있다. systems speedup을 넓게 주장할 경우 비교 필요. |
| P2 | **Parallelizing Tool Execution and LLM Generation for Low-Latency Agent Serving**, **2026-03-19**, 최신 제목. [논문](https://arxiv.org/abs/2603.18897) | tool 실행과 generation을 겹치는 serving 최적화 계열. 과거 검색 결과에는 다른 PASTE 제목이 남아 있어 최신 제목과 버전을 고정해야 한다. | 선택 비용과 tool wait의 구분을 배우되 동시 실험 확장 우선순위 낮음. |
| P2 | **CacheRouter: A Dual-Path Tool Routing Architecture with Cache-Preserving Main-Model Isolation for Long-Tail Tool Discovery**, **2026-08-24**. [논문](https://arxiv.org/abs/2608.22708) | 고정 core tools와 별도 long-tail routing 채널로 main prefix cache를 보존. 5개 고정 도구의 우리의 문제와 다르며 API pricing cache 이득과 local GPU latency는 동치가 아니다. | 고정 registry/hash는 지금 적용할 원칙. 수백 도구 검색 framework는 추가하지 않음. |

## 공개 decision model: 논문과 model card를 구분

| 모델/검증한 출처 | 확인 범위 | 적용과 한계 |
|---|---|---|
| [Kev 공식 저장소](https://github.com/jaredpalmer/kev), 2026-10-15 열람 | Qwen 계열 Jev-like choice/noul/score, temperature calibration. 특정 최초 release 날짜는 이번 검토에서 확정하지 않음. | 로컬 4B급 외부 decision baseline 후보. arbitrary API probabilities가 calibrated stopping quality를 보장하지 않음. image 지원은 checkpoint/runtime별 별도 확인. |
| [NeoHorse-Jev-4B 공식 model card](https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B), **2026-09 공개 자료** | prefill-only structured decision head, **text 또는 single image+text** 입력을 명시. bundle에는 별도 decision head와 matching runtime 필요. 정확한 최초 공개일은 이번 검토에서 확정하지 않음. | 'multimodal decision model 최초' 주장은 불가. 4B 추론 용량은 현실적이지만 우리 multi-image history와 동일 입력 지원이 아님. single-image subset external comparison 또는 capability restriction을 명확히 보고. |
| [GLiNER2.5-Decide 공식 model card](https://huggingface.co/fastino/GLiNER2.5-Decide), 2026-10-15 열람 | **340M text-only** classifier, caller-defined labels와 single-forward classification. 근간 GLiNER 논문과 이 checkpoint의 release를 혼동하지 않음. 정확한 최초 공개일 미확정. | 낮은 비용의 text-state routing baseline 후보. 이미지 내용을 title/caption으로 바꾸면 pixel-input 정책과 입력 정보가 달라짐; text-only ablation에서만 동일 조건 비교. |

현재 6-action head와 비교할 때 가장 공정한 모델 baseline은 **같은 Gemma의 생성 planner / 첫 토큰 readout / 동일 hidden linear 또는 MLP head**다. 외부 모델은 모델 능력·입력 형식까지 바뀌므로 보조 비교다. 96GB라는 메모리 용량만으로 BF16 training·long-context KV·배치·실행시간이 보장되지 않는다. 실제 footprint 측정 후 순차 실행한다.

## 논문별로 배울 것: 코드 점검과 별도 실험을 분리

| 출처 | 전이할 설계 원칙 | 현재 코드에서 확인할 항목 | 별도 실험이 필요한 항목 / 우선순위 |
|---|---|---|---|
| ToolkenGPT | 선택과 argument realization을 분리 | `DirectHFReader`가 선택된 이름을 코드로 만들고 args schema를 strict validate하는지 | tool-token/label-row vs 독립 head, same-backbone·same-data / **P0** |
| SII-Decision | pretrained label readout은 강한 무학습 기준선; 로컬 정확도와 시스템 성공률 분리 | 실제 legal 6 labels tokenizer/context 정합, full trajectory latency | first-token6 vs trainedhead6, 품질·p95·calibration / **P0** |
| LatentGate | representation geometry와 pooling도 baseline | final norm last state 사용, gold/source identifier가 feature에 없는지 | meanpool/whitening을 train-only fit, same-budget probe / **P0** |
| Stop-RAG | stop은 미래 retrieval benefit의 선택 | 강제 FINAL과 learned FINAL 구분, remaining budget 기록 | generic value-only6 vs proposed supervision; binary STOPCONT는 보조 / **P0** |
| 2026 structured-stop 연구 | 첫 STOP이 실제 배포 outcome을 정함 | 잘못된 early stop 후 회복 기회를 허용하지 않는지 | source-group heldout rollout, step error accumulation / **P0** |
| GLiNER/Kev/NeoHorse | 모델 인터페이스와 실제 입력 능력을 고정 | text/single-image/multi-image capability, checkpoint/revision/runtime hashes | 호환 subset에서 외부 baseline; 임의 modality 제거 금지 / **P1** |
| Luna-2 | head/adapter와 shared backbone 비용 분리 | head ms만 보고하지 않고 full prefill/args/answer 포함 | head only vs adapter는 genericQ baseline 이후 / **P1** |
| Router-R1/SkillOrchestra | outcome와 cost를 공동 고려 | 도구 latency와 모델 latency를 분리 기록 | offline cost-aware Q부터; fullRL/routerpool 확장 보류 / **P1** |
| LAR | 실제 행동 horizon과 wrapper token 수를 구분 | directdispatch가 query/arg 생성까지 없앴다고 주장하지 않는지 | 다단계 latent actions는 새로운 학습·실험 범위 / **P2** |
| SMC/parallel serving | 직렬 critical path를 재야 함 | controller·arguments·toolwait·answer 순서와 walltime | 동일 GPU에서 speculation contention 포함 계측 / **P2** |
| CacheRouter | 정확한 prefix identity가 cache 재사용 조건 | registry hash/schema/instruction/runtime 일치 | KV 최적화는 우선 correctness parity부터 / **P2** |

이 체크리스트는 모두 채택하라는 뜻이 아니다. **P0 비교와 현재 코드 정합을 먼저 끝내고**, 한 가설을 지지하는 증거가 생길 때만 P1/P2를 추가한다.

## RAG 관련 근접 기반과 벤치마크

- [**A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces**, 2026-02-03](https://arxiv.org/abs/2602.03442): 세 가지 hierarchical retrieval interfaces가 기본 비교 기반. 현재 5-tool G1 변형을 원본 A-RAG와 똑같다고 부르지 않고, 동일 5 tools를 받는 생성형 controller를 최우선 대조한다.
- [**MultiModalQA: Complex Question Answering over Text, Tables and Images**, 2021-04-13 arXiv/ICLR 2021](https://arxiv.org/abs/2104.06039): 현재 multi-hop multimodal pilot에 잘 맞는다. public table과 gold evidence scope의 경계를 지키고 source/entity 연결로 split을 묶는다. 오래된 데이터라는 이유로 부적합하지 않지만 단일 dataset 결과만으로 일반화 주장 금지.
- [**ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents**, 2025-02-25](https://arxiv.org/abs/2502.18017): ViDoSeek benchmark와 탐색·요약·성찰 기반 visual iterative workflow. 멀티모달 iterative RAG 자체도 새롭지 않다는 직접 대조. 전체 multi-agent 재현보다 같은 corpus/reader의 고정 retrieval 및 생성 agent 비교부터.
- [**Benchmarking Retrieval-Augmented Multimomal Generation for Document Question Answering**, 2025-05-22 arXiv 제목](https://arxiv.org/abs/2505.16470): MMDocRAG는 cross-page/cross-modal evidence와 quote selection 평가를 제공한다. heldout transfer 후보지만 answer/quote output task를 우리 span JSON task와 맞추는 adaptation 비용을 명시해야 한다.
- [**Improving Passage Retrieval with Zero-Shot Question Generation**, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.249/): UPR는 passage-conditioned query likelihood reranking이다. relevance baseline이지 tool-action value나 stop probability가 아니다. known-token teacher forcing은 sampling이 없어도 query-document contextual compute를 요구한다.

## 판단

**현재 full A-RAG direct head는 비교 가능한 실험 플랫폼이다. 그 구조 자체의 독창성은 약하다.** 특히 ToolkenGPT, SII-Decision, LatentGate, Stop-RAG가 각각 선택/인자 분리, first-token readout, frozen-head routing, value stopping을 직접 선점한다. multimodal 확장도 NeoHorse-Jev와 visual iterative RAG가 있어 독립 신규성으로 부족하다.

아직 방어 가능한 후보는 '증거가 최종 답변과 후속 acquisition에 미치는 서로 다른 효과를 학습하면, 일반 outcome-Q가 구분하지 못하는 tool choice를 실제 5-tool trajectory에서 회복하는가'라는 **supervision과 배포 효과의 좁은 가설**이다. 이는 새로 입증할 대상이며 현재 성과가 아니다. 같은 rollout/data/capacity/cost의 genericQ와 비교해 개선하지 못하면 기여에서 제외해야 한다. directdispatch는 그 가설을 실행·측정하는 수단으로 두는 것이 타당하다.

## 追加: 품질과 메커니즘을 중심으로 재정의 (사용자 최신 우선순위)

도구 호출 형식과 ARAG 연결은 연구 기여가 아니라 구현 기반이다. 목표는 **왜 decision 학습이 실제 agent 선택을 더 좋게 만드는지**를 밝히고, RAG 밖에서도 그 원리가 성립하는지 검증하는 것이다. 아래가 이 방향의 더 가까운 경쟁 연구다.

| 최신 1차 출처 | 실제 근접성 / 배울 원칙 | 지금 확인 / 추가 실험 |
|---|---|---|
| [**MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains**, arXiv 2603.00873, 2026-03](https://arxiv.org/abs/2603.00873) | answer evidence removal의 utility와 downstream entity navigation을 구분하는 HAVE. '답변에 안 쓰여도 후속 탐색에 유용한 evidence'라는 현상 자체는 새롭지 않다. | 현상 최초 주장 금지. answer-only removal label·entity-navigation proxy를 proposed labels와 같은 데이터에서 비교 / **P0** |
| [**ToolAnchor: Anchoring Counterfactual Context to Boost Agentic Tool-use Capability**, 2026-07-14](https://arxiv.org/abs/2607.14145) | teacher가 counterfactual context를 제안하고 student rollout으로 검증한 뒤 성공 개입을 post-training. counterfactual context→좋은 tool use 학습 역시 이미 연구됨. | 우리의 조건은 원 evidence의 planner/executor 접근을 교차 통제한다는 점; teacher-written context가 아님. 단순 evidence perturbation/augmentation과 분리 비교 / **P0** |
| [**Counterfactual Tool Ranking under Utility, Cost, and Privilege Constraints**, 2026-09-19](https://arxiv.org/abs/2609.22819) | direct/DR outcome regression, propensity/support gap, BFCL-derived tool ranking. 'counterfactual utility로 tool을 선택'도 충분한 신규성이 아님. | 완전 branch rollout이면 DR 필요 없음. 지원 없는 action을 추정으로 꾸미지 말고 matched full-return regression을 가장 강한 기준선으로 설정 / **P0** |
| [**Value of Information: A Framework for Human-Agent Communication**, 2026-01-10](https://arxiv.org/abs/2601.06407) | clarification expected utility gain과 user cost를 비교. information value와 confidence를 구분해야 함. | head 확률을 VoI라 부르지 말고 continuation return difference로 평가 / **P1** |
| [**Rational Clarification by Assistive Agents via Value-of-Information Reasoning**, 2026-09-29](https://arxiv.org/abs/2609.37588) | REVOIR는 clarification의 expected task-reward improvement를 추론하며 QA와 household planning에 적용. task transfer+VoI만으로도 신규성 부족. | 우리의 차이는 inference-time reasoning이 아니라 interventional supervision의 역할. 독립 generic-Q보다 질이 좋아지는지 필수 검증 / **P1** |
| [**ToolComp: A Multi-Tool Reasoning & Process Supervision Benchmark**, 2025-01](https://arxiv.org/abs/2501.01290) | multi-tool reasoning의 process supervision benchmark. intermediate labels 추가 자체는 기존 범주. | auxiliary를 '과정 supervision 최초'라 주장 금지. 무작위/permuted/equal-size generic auxiliary로 signal-specificity 확인 / **P1** |

### 추천하는 하나의 좁은 가설

**정보를 읽은 결과의 직접 실행 가치와 다음 정보 수집을 바꾸는 가치를 분리해서 감독하면, 행동 성공 여부만 학습한 동일 규모 decision model보다 숨은 탐색 가치를 더 잘 보존하고 전체 task 성공을 높인다.** 적용 대상은 'observation→next action→task outcome'을 갖는 agent이며 RAG만으로 한정하지 않는다. 이 문장은 가설이고 새 방법의 확정 주장이 아니다. MC-Search와 ToolAnchor를 넘어서는 차이는 교차 접근 label이 같은 예산의 일반 outcome supervision보다 어떤 상황에서 실제로 더 잘 일반화하는지에 달려 있다.

RAG에서는 query planner와 answer reader의 접근 `P/E∈{0,1}`를 교차한다. RAG 밖에서는 **WebShop의 정보 획득 단계(검색/상품 조사)와 최종 상품 선택 단계**에 같은 접근 경계를 둔다. WebShop은 텍스트 기반 shopping 환경으로 별도 agent family이며 [공식 논문, 2022-07-04](https://arxiv.org/abs/2207.01206)이 있다. 결제·구매 같은 외부 부작용은 시뮬레이터 안에서만 수행한다. 현재 G1의 tool action vocabulary를 WebShop에 억지로 공유하지 않고, 공통 hidden adapter/효과 학습 objective를 유지하며 task별 legal-action head를 둔다. 이는 action ID 자체의 zero-shot transfer가 아니라 **학습 원리의 cross-family transfer**다. 같은 trunk를 직접 zero-shot 공유하려면 별도 hypothesis로 사전에 고정해야 한다.

우선 RAG에서 메커니즘을 확인하고 WebShop에서는 preregistered protocol을 그대로 적용한다. 새 framework나 memory lifecycle·latent action·speculation을 동시에 추가하지 않는다. WebShop이 정보획득과 실행 경계를 독립적으로 구현하기 어렵거나 baseline reader가 task를 풀지 못하면, transfer 검증에 부적합하다고 판단하고 환경을 다시 선정한다.

### 무엇이 '왜 좋아지는가'에 대한 증거인가

1. 같은 원 observation과 동일 rollout budget에서 4 access cells를 생성한다. effect label은 **실제로 실행한 개입의 conditional contrast**이며 prompt 경계·budget·결과의 matching을 고정한다. 이를 자연 세계의 보편적인 causal effect라고 확장하지 않는다.
2. heldout 상태에서 '최종 실행에는 직접 도움 적음, 다음 탐색에는 도움 큼'인 bridge evidence와 direct-only evidence를 사전 정의한 contrast로 구분한다. 일반 relevance와 F1 confidence로 대신 정의하지 않는다.
3. outcome-Q, outcome-Q+generic auxiliary, proposed auxiliary의 action regret와 end-to-end success를 같은 capacity/data/optimizer에서 비교한다. effect 예측이 좋아도 action이나 성공이 안 좋아지면 기여가 입증되지 않는다.
4. planner 접근을 차단했을 때 bridge evidence의 이득이 사라지고, irrelevant evidence·permuted contrast labels에서는 학습 이득이 사라지는지 검증한다. attention 그림이나 head weight 시각화만으로 메커니즘을 주장하지 않는다.
5. RAG에서 고른 objective와 weighting을 WebShop test outcome에 맞춰 재조정하지 않고 평가한다. source/task-family leakage를 막고 각 family의 strongest generated planner 및 genericQ와 비교한다.

4-cell 수집 자체가 supervised signal을 더 많이 주는 효과일 수 있으므로 generic-Q 대조군에도 **같은 모든 cell outcome과 같은 rollout 수**를 제공해야 한다. proposed model만 추가 cell을 보고 baseline이 11-cell만 본다면 분해의 기여를 식별할 수 없다. full tool registry의 모든 행동을 효율적으로 branch하기 어려우면, 소수의 관측 가능한 정보획득 decision boundary부터 pilot로 범위를 정하되 이를 전체 5-tool 학습으로 과장하지 않는다.

**반증:** 두 family에서 improvement가 나타나지 않거나, effect decomposition 대신 같은 outcome들을 generic auxiliary로 학습해도 같은 개선이 나타나거나, 추가 데이터/용량만으로 설명되면 제안 supervision의 독자적 가설을 기각한다. 이 실패 가능성을 유지해야 '더 빠른 routing'을 넘어 품질과 작동 원리를 연구하는 논문이 된다.
