# 비싼 시각 증거 획득을 제어하는 연구 방향

검토일 2026-10-15 UTC. 원 논문·저자 코드·공식 dataset card를 확인했다. 연구 설계이며 실행 결과·독창성 확정·ICML 채택 예측이 아니다. 기존 폐기된 실험 결과를 사용하지 않는다.

## 제안: 새로운 head보다 “추가 시각 관측의 가치”를 중심에 둔다

핵심 후보는 **현재 관측에서 어느 후보를 더 비싼 fidelity로 읽어야 다음 행동과 최종 성공이 실제로 개선되는가를 학습하는 제어 원리**다. tool-name decoding 제거는 실행 최적화로 두고 기여의 중심에서 제외한다. 멀티모달 retrieval은 후보를 제공하며, controller는 관련성 순서와 다른 **추가 읽기의 조건부 효용**을 판단한다.

처음에는 `현재 근거로 계속/답변`, `후보 i를 낮은 fidelity로 열기`, `이미 본 후보 i를 높은 fidelity로 재읽기`처럼 사전 고정된 두 fidelity만 사용한다. 새 crop generator, vision token pruning, memory framework, retriever 공동학습까지 동시에 만들지 않는다. “where”는 기존 shortlist의 동적 pointer로 한정한다. 임의 bbox/zoom은 효과 확인 뒤 별도 확장이다.

검증할 차이는 **관련하지만 이미 충분히 읽은 페이지**, **관련성은 낮아도 다음 검색·조작을 가능하게 하는 bridge**, **작은 글씨 때문에 추가 해상도만 도움이 되는 페이지**를 비용과 함께 구분하는가이다. 단순 `Q(s,a)`가 이미 충분하면 새 방법이 필요하지 않다. 제안 supervision이 필요한 이유는 실패 포함 자연 상태에서 generic Q보다 어떤 구분을 더 잘 학습하는지로 입증한다.

## 비용을 나눠야 잘못된 가속 주장을 피한다

1. **Retrieval index**: ColPali/ColQwen 계열의 페이지 patch embeddings와 late interaction 검색. 페이지를 미리 encode하는 비용·저장량이 있다.
2. **Reader vision encoding**: 선택된 원 이미지·페이지·crop을 Reader의 vision encoder로 처리한다. retrieval embedding이 Reader image token을 그대로 대체하지 않는다.
3. **Reader prefill**: 전달된 visual tokens와 누적 텍스트가 Transformer 문맥에 들어간다. vision encoder output cache가 있어도 매번 이 비용까지 사라지는 것은 아니다.
4. **Decode와 control**: query/인자·답변 생성과 행동 readout. 생성하지 않는 head도 전체 멀티모달 prefill을 실행하면 큰 비용을 이미 지불했다.

[ColPali 원 논문](https://arxiv.org/abs/2407.01449)과 [공식 ColVision 구현](https://github.com/illuin-tech/colpali)은 page-image 기반 multi-vector retrieval의 강한 기준이다. [VisRAG](https://arxiv.org/abs/2410.10594)와 [공식 코드](https://github.com/OpenBMB/VisRAG)는 parsing-free visual RAG를 제시한다. 현재 저장소에는 multi-image evidence reasoning을 다루는 EVisRAG도 공개돼 있다. [M3DocRAG/M3DocVQA 원문](https://openaccess.thecvf.com/content/ICCV2025W/Findings/papers/Cho_M3DocVQA_Multi-modal_Multi-page_Multi-document_Understanding_ICCVW_2025_paper.pdf)은 retrieval encoder와 생성 VLM의 visual encoder를 구분한다. 따라서 이미 만들어진 ColPali vectors를 Gemma image features처럼 넣는 가속은 동치 최적화가 아니다.

## 가까운 선행연구: 넓은 개념 대부분은 이미 존재한다

| 확인한 원문/공식 구현 | 이미 다루는 부분 | 우리 연구가 넘어야 할 경계 |
|---|---|---|
| [V* / SEAL, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_V_Guided_Visual_Search_as_a_Core_Mechanism_in_Multimodal_CVPR_2024_paper.html), [코드](https://github.com/penghao-wu/vstar) | 고해상도 이미지에서 필요한 세부 영역을 찾는 guided visual search | “필요한 곳만 확대한다” 자체는 기여가 아님. 자연 사진의 작은 객체와 문서 OCR/표 판독은 task가 다름 |
| [Doc-V*, 2026](https://arxiv.org/abs/2604.13731), [공식 코드](https://github.com/SeerRay-Lab/Doc-V) | thumbnail overview→semantic retrieval→targeted page fetching, expert imitation와 RL | 저해상도 개요·선택적 고해상도 페이지·순차 evidence aggregation의 조합은 이미 있음 |
| [InSight-doc, 2026-08](https://arxiv.org/html/2608.10628v1), [공식 코드](https://github.com/m-Just/InSight-doc) | 전체 저해상도 문서에서 region zoom을 반복하는 learned policy; 해상도를 reasoning-time 자원으로 취급 | **가장 직접적인 novelty threat.** adaptive fidelity·zoom policy·accuracy/latency frontier를 처음 제안했다고 할 수 없음. retriever-free라는 차이만으로 독창성 충분하지 않음 |
| [TAP-RAG, 2026-07](https://arxiv.org/html/2607.18917v1) | task prior에 따른 evidence policy와 선택적 page-image inspection, text/visual/structural fusion | “무엇을 언제 시각적으로 읽을지”라는 포괄 주장 중복. 동일 후보/예산의 learned marginal-value supervision 비교가 필요 |
| [MAGE-RAG, 2026-06](https://arxiv.org/html/2606.15906) | page–element evidence graph 위에서 budgeted activate/open/search/prune를 수행하고 compact multimodal context 생성 | **직접적인 중복 위험.** adaptive granularity·선택적 materialization·stateful evidence controller라는 조합도 기존 연구. 그래프를 새로 추가하는 것으로 차별화하지 않음 |
| [FastV](https://arxiv.org/abs/2403.06764), [공식 코드](https://github.com/pkunlp-icler/FastV) | early attention 기반 후속 visual token pruning | visual token 수 축소 자체가 novelty 아님. masking 실험과 실제 token dropping latency를 구분. Gemma 호환성은 별도 구현/검증 필요 |
| [LLaVA-UHD](https://arxiv.org/abs/2403.11703), [현 공식 v3 코드](https://github.com/thunlp/LLaVA-UHD), [v4 코드](https://github.com/THUMAI-Lab/LLaVA-UHD-v4) | 고해상도 slicing·compression, 최근 intra-vision-encoder 압축 | reader architecture·encoder 재학습 축이다. 한 GPU 프로젝트에서 새로운 vision encoder 학습과 controller 기여를 동시에 하지 않음 |
| [vLLM encoder cache 문서](https://docs.vllm.ai/en/v0.15.0/api/vllm/v1/core/encoder_cache_manager/) | multimodal encoder output caching | pixel cache나 vision encoder bypass 자체가 새 방법 아님. 반복 요청 cache 이득을 선택 정책의 학습 이득으로 섞지 않음 |
| [MC-Search 공식 프로젝트](https://mc-search-project.github.io/), [논문](https://openreview.net/forum?id=JEGDp1E4OH) | cross-modal 순차 탐색·structured subquestion chains | “한 관측이 다음 검색을 돕는다”는 문제도 선행이 있음. image fidelity의 추가 가치가 독립적으로 필요한지 검사 |
| [MMSearch-R1 공식 코드](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1) | RL 기반 on-demand multi-turn text/image search | 언제/어떻게 search할지 RL로 학습하는 넓은 아이디어 중복. live search와 고정 local-corpus fidelity 제어는 분리 |

이 표는 exhaustive novelty audit가 아니다. 특히 active information acquisition, budgeted sensing, adaptive computation과 cost-sensitive decision theory의 고전 선행까지 방법을 정한 뒤 추가 확인해야 한다. “작은 head + visual VoI”만으로 새로운 고영향 방법이라고 확정할 수 없다.

### 논문 기여가 될 수 있는 좁은 가설

> 저비용 관측을 얻은 뒤 그 관측이 **후속 행동 선택을 개선하는 효과**와 **이미 알려진 정답 근거를 더 선명하게 읽는 효과**가 다른 상태를 계측하고, 그 구분을 학습하는 supervision이 동등 수집 예산의 generic value learner보다 새로운 source/task에서 비싼 시각 관측을 더 잘 배분한다.

아직 입증되지 않은 가설이다. 추가 fidelity의 utility는 원본·저해상도·고해상도를 모두 본 teacher의 사후 정보이므로 배포 controller 입력에 넣으면 안 된다. 후보 low/high의 return 차이를 회귀하는 것만으로 새로운 알고리즘이라고 하지 않는다. 선택 이후의 정보 획득이 다음 행동과 성공에 미치는 차이를 일반화시키는 학습 구조가 추가 가치가 있는지 보여야 한다.

## 실제 벤치마크 적합성

고해상도 제어 연구에서는 **페이지 이미지**와 **문서 안의 자연 사진**을 구분한다. 이미지 metadata title만 읽어 푸는 QA, 표를 텍스트로 이미 제공한 QA, caption만으로 풀리는 QA는 pixel-fidelity 기여의 직접 증거가 아니다.

| 후보 | 원본·검색 corpus·근거 | 고해상도/다단계 적합성 | 접근/라이선스/측정 |
|---|---|---|---|
| **M3DocVQA** | [공식 코드](https://github.com/konnisan/m3docvqa), [공개 HF PDFs·annotations](https://huggingface.co/datasets/ad6398/m3docvqa). MMQA 계보 QA와 3,368 PDFs/약41K pages | 큰 page corpus의 multi-document 검색에 적합. 단, 질문 일부는 MMQA의 사진/텍스트 중심으로 고해상도 문서 detail이 필수인지 개별 audit 필요 | 코드/HF card Apache-2.0; 원 Wikipedia 내용 권리 별도. official score·문서/page grounding 재현. 원 MMQA support doc를 gold page bbox와 동일시하지 않음. 기존 pipeline 재사용 가능성이 가장 큼 |
| **VisDocAgentBench, 2026-08** | [공식 프로젝트](https://hulx2002.github.io/VisDocAgentBench/), [논문](https://arxiv.org/html/2608.17889), [공식 코드](https://github.com/hulx2002/VisDocAgentBench), [HF](https://huggingface.co/datasets/hulx2002/VisDocAgentBench). 100 scientific docs/2,375 rendered pages/120 unique-target queries | direct/one-bridge/two-bridge 각40으로 **navigation 효용의 mechanism diagnostic에 가장 직접적**. 고정12-action budget, page inspection·crop·search, 최종 top10 page ranking. QA benchmark가 아님 | 코드 Apache-2.0, benchmark-authored annotations CC-BY-4.0, 원 page license 별도. 실제 배포1,469 pages와 fullmanifest, 나머지 exact arXiv version에서144DPI 재구성. 120Q를 대형 학습/최종 단독test로 사용하지 않음 |
| **ViDoRe v3** | [공식 소개](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3), [benchmark code](https://github.com/illuin-tech/vidore-benchmark), [예: HR corpus](https://huggingface.co/datasets/vidore/vidore_v3_hr). corpus images/PDFs·query/qrels 제공 | 다양한 시각 문서 retrieval 및 pipeline evaluation. qrels 기반 relevance 평가이므로 **retrieval-only reference**; bridge depth나 QA 최종 성공의 대체물이 아님 | HR card CC-BY-4.0, 각 subset별 권리 별도 확인. public corpus/PDF listing 확인. static retriever와 agentic pipeline의 retrieval quality/cost 비교에 적합 |
| **ViDoSeek** | [공식 HF](https://huggingface.co/datasets/autumncc/ViDoSeek), [ViDoRAG](https://github.com/Alibaba-NLP/ViDoRAG). 실제 PDF ZIP와 `query`, `reference_answer`, `reference_page`, source_type, query_type 공개 | **visual fidelity 진단에 특히 적합**: text/table/chart/2D-layout, single/multi-hop 구분. 작은 표/배치 detail과 page retrieval을 함께 검사 가능 | HF card Apache-2.0. 페이지 번호 convention을 loader에서 시각 확인하고 bbox gold를 가정하지 않음. 제공 split 성격을 확인하고 train/test 재구성은 새 adaptation으로 명명 |
| **MC-Search** | [공식 HF](https://huggingface.co/datasets/YennNing/MC-Search): parquet images·QA·subqa_chain, 14,037 passages/10,953 image metadata KB | modality transition·multi-step search에 강함. corpus가 자연/지식 이미지 중심이면 **고해상도 문서 확대** 주장의 주평가로 약할 수 있음 | QA/중간 답/reference chain은 있으나 optimal tool policy가 아님. HF card license 미명시. KB 모든 image bytes·source URL completeness를 먼저 확인. caption-only shortcut·gold images 입력 경계 통제 |
| **MultiModalQA** | [공식 데이터](https://github.com/allenai/multimodalqa): text/table/image corpus, supporting_context | text↔image bridge의 controlled 진단에는 좋음. Wikipedia photos와 public table이 있어 native high-res document detail 연구의 주 benchmark로 부족 | 전체 image ZIP 공개. 포괄 dataset LICENSE 미명시. 공개 table_id 배경 유지, intermediate_answers와 gold subset은 private. official list EM/F1 사용 |
| **MADQA, 2026** | [공식 코드](https://github.com/OxRML/MADQA), [HF](https://huggingface.co/datasets/OxRML/MADQA), [논문](https://arxiv.org/abs/2603.12180): 800 heterogeneous PDFs/2,250 human-authored Q | 실제 여러 PDF의 page search·visual reasoning에 잘 맞는 외부 검증 후보 | 코드 Apache-2.0이나 dataset card CC-BY-NC-4.0, HF asset 요청이 인증/접근 제한 응답. 접근 확보 전 일정에 의존하지 않음. answer·Page F1 등 official eval, gold pixel region을 가정하지 않음 |
| **VisBrowse-Bench, 2026** | [공식 HF](https://huggingface.co/datasets/Zhengbo-Zhang/VisBrowse-Bench), [코드](https://github.com/ZhengboZhang/VisBrowse-Bench): 169 visual-native browsing Q | web image/text 수집과 cross-validation 평가. 정적 PDF corpus/fidelity task와 다른 환경 | question/answer 암호화와 공식 decrypt script가 명시됨; card license 미명시. live web/search API·원문 변경·재현 비용이 있어 첫 주평가로 비추천 |

이번 확인은 원문·card·file listing 검토이며 새 전체 PDF corpus나 image collection을 다운로드해 완전성을 검증한 것은 아니다. 축소 corpus를 만들면 동일한 shared pool과 distractor 정책을 공개하고, gold를 문항마다 다르게 선택해 “retrieval”이라고 부르지 않는다.

### 갱신한 최소 benchmark 역할 분담

- **주 학습/QA 환경 하나:** M3DocVQA 또는 ViDoSeek. 고해상도 문서 근거를 실제 읽어야 하는지로 선택한다.
- **원인 진단 하나:** VisDocAgentBench의 direct/one-bridge/two-bridge 고정 strata. bridge가 늘 때 첫 관측이 다음 후보 발견·검사·순위 결정에 주는 효과를 분리한다. 원래120Q를 결과에 따라 hard subset으로 다시 골라 prevalence를 주장하지 않는다. 동일문서 관련 query 간 dependence도 기록한다.
- **선택적 외부 retrieval 검증:** ViDoRe v3. 후보 재탐색/rewriting의 가치가 static scoring 이상인지 검사한다. QA·최적 행동 label·multihop chain을 제공한다고 가정하지 않는다.
- **비RAG 전이 하나:** 앞선 문서의 stateful local API 환경. 시각 fidelity claim과 공통 decision principle transfer를 분리한다.

이 네 역할은 전부 즉시 실행할 약속이 아니다. 먼저 주 QA 환경과 작은 mechanism diagnostic을 고정하고, 공통 원리가 통과할 때 nonRAG transfer를 수행한다. ViDoRe v3는 추가 여력이 있을 때 사용한다.

VisDocAgentBench 공식 저장소는 page bytes의 재배포 경계를 명시하고 missing source pages의 deterministic reconstruction을 제공한다. 전체 corpus가 준비되지 않은 채 subset 점수를 full120 official 결과로 표시하지 않는다. 공개 gold target/support annotations는 evaluator 전용이고 opaque handles를 사용한다. 원 baseline planner는 매우 큰 Qwen3.5-397B-A17B checkpoint를 사용하는 설정도 포함하므로 그 전체 재현을96GB 한 대의 필수 조건으로 삼지 않는다. 동일 benchmark를 로컬 작은 frozen reader로 실행하면 그 adaptation과 native reference를 구분한다.

[NVIDIA 공식 ViDoRe v3 agentic pipeline](https://github.com/NVIDIA/NeMo-Retriever/blob/main/retrieval-bench/submissions/vidore_v3_agentic_pipeline.md)은 repeated retrieve/query rewriting과 최종 reranking, native commercial planner, late-interaction retriever를 결합한다. 따라서 “iterative retrieval 자체가 도움이 됐다”도 새 기여가 아니다. 문서의 과거 leaderboard rank를2026-10-15 최신 순위로 재인용하지 않는다. 원 구성 전체를 재현하지 않을 경우 공통 backbone/후보/예산으로 옮긴 adaptation으로 평가하며, 원본 score와 controlled 성능을 혼합하지 않는다.

**권고 순서:** M3DocVQA 기존 인프라에서 pixel-sensitive 상태가 충분한지 먼저 확인한다. 부족하면 주 task를 ViDoSeek로 한 번 교체한다. ViDoSeek와 M3DocVQA를 동시에 대형 구축하지 않는다. MC-Search는 다음 검색에 주는 효과의 external probe로 적합하나 고해상도 문서 결과를 대신하지 않는다. MADQA는 접근이 확보됐을 때 외부 검증으로 사용한다.

## 공정한 최소 실험

동일 candidate pool·Reader·fidelity levels·이미지/call/context 예산에서 다음을 비교한다.

- 고정 top-K high fidelity, 고정 low fidelity, low→high rank-order escalation.
- 현재 답변 confidence 또는 retrieval score threshold, OCR character-size/page density, 중복/coverage 기반 heuristic. 어떤 heuristic도 gold page/정답 bbox를 입력으로 받지 않음.
- 같은 backbone의 생성형 controller, constrained first-token, BC, generic action-value learner.
- generic Q + 같은 paired low/high training states; 제안 auxiliary; 같은 전체 teacher GPU-hour의 추가 natural branch.
- 가능하다면 Doc-V*/InSight-doc/TAP-RAG 중 **실제 공개 checkpoint·코드가 공통 task에 맞는 하나**를 native protocol reference로 추가. 논문 숫자를 같은 표의 controlled head-to-head 결과로 섞지 않음.

고정 후보 pool에서 먼저 fidelity 선택의 효과를 분리하고, 그다음 closed-loop 검색을 켠다. 전자는 selection diagnostic이며 후자를 대신하지 않는다. crop이 필요한 경우 사전에 고정한 grid/layout crop만으로 시작하고, crop proposals를 제안에만 유리하게 제공하지 않는다. 큰 해상도에서 이미지를 처리한 뒤 그 결과로 “고해상도 읽을지”를 정하면 비용을 선지불한 정책이다.

정답 EM/F1/ANLS·공식 evidence metrics와 함께 **선택한 action의 실제 utility, 무익한 high-res 읽기 비율, 필요한 detail을 놓친 종료, 다음 검색/조작 성공률, observed branch regret**를 기록한다. pixel removal·동일 크기 distractor·OCR/caption-only·low/high 대조로 원인을 확인한다. 작은 영역이 정답인 sample만 골라 전체 분포의 결과라고 주장하지 않는다.

## Cache와 단일 GPU 실행의 범위

동일 pixels·model revision·processor·resize/crop/patch layout·dtype의 **vision encoder output**는 architecture가 허용하면 cache할 수 있다. crop/fidelity가 바뀌면 별도 key다. query-independent output임을 확인하고 모델별 image-position metadata를 포함한다. 이는 각 요청의 context-dependent LLM KV cache와 다르며, retrieval vectors와도 다르다. prompt 위치·history가 달라지면 기존 KV 재사용은 동치가 아닐 수 있다.

cache를 쓰면 모든 정책에서 같은 cold/warm 규칙, 동일 warmup, 원본/축소/crop 캐시 저장·적재 비용을 적용한다. offline로 모든 high-res features를 precompute한 경우, online latency와 전체 preprocessing amortization을 함께 보고한다. 선택되지 않은 모든 이미지의 비싼 Reader feature를 미리 계산하는 정책은 “읽기 비용을 회피했다”는 claim과 다르다.

96GB GPU에서는 frozen reader 하나·retriever 하나·CPU head 학습과 수백 문항 branch collection으로 시작할 수 있다. 다만 가능성 판단일 뿐 throughput 보장은 아니다. 40K 페이지의 multi-vector index는 CPU RAM/disk도 고려해야 하며, index GPU 상주를 기본 가정하지 않는다. 측정 분해는 image load/decode→resize→vision encoder→LLM prefill→controller→query/argument decode→retrieval→final answer다. 시각 비용이 어디서 발생하는지 계측한 뒤 절감 상한을 계산한다. 실제 token dropping 없는 masking을 가속으로 세지 않는다.

## 비RAG 일반화와 최종 진행 판단

사용자가 요구한 agent 일반성은 유지한다. 원리는 **비싼 관측을 추가로 얻으면 후속 행동이 얼마나 개선되는가**다. 비RAG stateful API에서는 coarse status→detailed inspection 같은 실제 서로 다른 정보/비용이 있는 tool pair가 필요하다. 이미 공짜로 얻은 완전한 state에서 텍스트를 임의로 숨겨 비싼 sensor라고 꾸미지 않는다. 앞선 [비교군 검토](JEV_RELATED_BASELINES_20261015.md)의 BFCL local engine은 action-quality transfer에는 적합하지만 실제 이미지 fidelity 비용은 없다. 따라서 그 결과는 **제어 학습 원리의 transfer**, 고해상도 가속 검증은 visual document task로 명확히 구분한다.

진행 조건은 (1) high fidelity가 도움이 되는 자연 상태와 낭비되는 상태가 모두 존재하고, (2) 읽기 전에 얻을 수 있는 합법적 cheap feature로 구분 가능하며, (3) 강한 generic Q와 고정/heuristic escalation보다 실제 선택과 최종 성능이 개선되고, (4) 새로운 source/task에서도 그 이유가 개입 분석으로 유지되는 것이다. 하나라도 실패하면 단순 head·zoom·cache를 합쳐 새 기여로 포장하지 않는다. 현 시점의 가장 강한 제안은 이 **가치 구분을 위한 학습 신호**를 좁게 설계·반증하는 연구이며, 새로운 vision architecture 전체를 만드는 안이 아니다.
