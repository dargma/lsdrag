# Direct dispatch decision head: 설계 검토

작성 2026-10-15. 현재 실행 중인 v1.1 수집을 변경하지 않은 후속 연구 설계다. 아래 속도·성능은 측정 결과가 아니라 비용 분해와 검증할 가설이다. 신규 모델 실행·학습은 수행하지 않았다.

## 1. 현재 구현에서 실제로 절약하는 것

`src/lsdrag/hf_reader.py`의 v1.0 head는 full multimodal prefill의 마지막 norm 표현으로 행동을 고른다. 이어 `call:semantic_search{` 등 도구 이름과 구조 prefix를 강제로 넣고 같은 KV에서 인자를 생성한다. 따라서 절약 대상은 도구 선택/이름/구조의 자율 생성이며, 검색어·인자 생성은 남는다. 강제 prefix도 모델 forward 비용은 발생한다. 최종 답변 내용 생성 역시 남는다. 작은 head가 붙었다고 full backbone 비용이 없어지지는 않는다.

현재 `scripts/run_v11_pilot.py`는 더 명확하게 다음을 분리한다.

| 단계 | 현재 방식 | head로 제거되는가? |
|---|---|---|
| 결정 상태 표현 | 질문+증거를 full Gemma VLM prefill | 아니오 |
| A/B/C 행동 | first-token 또는 MLP Q3 | 토큰 decode는 생략 가능; first-token도 동일하게 가능 |
| 추가 검색의 query | 별도 prompt prefill+최대 48 tokens 생성 | 아니오 |
| query embedding/index search | 별도 embedding runtime+vector search | 아니오 |
| 최종 답변 | 별도 prompt prefill+최대 128 tokens 생성 | 아니오 |

현재 head 경로의 acquire iteration은 개략적으로 `P_decision + H + P_query + D_query + E_query + R + P_answer + D_answer`다. answer-now이면 query 관련 항을 생략한다. 다른 prompt 사이 KV 재사용은 구현되어 있지 않다. JSON/tool prefix만 제거한 효과를 전체 의사결정 제거 효과로 설명하면 안 된다. 기존 30.6% 결과로 이 비용 항들의 인과적 기여를 분리할 수 없다.

## 2. 좁은 후보 세 가지

### A. Full-state joint pointer (먼저 구현할 정확한 대조군)

한 번 full VLM state를 만든 뒤 head가 `(action_enum, candidate_ID/slot)`를 고르고 Python이 곧바로 함수를 호출한다. `READ_TEXT(id)`, `READ_IMAGE(id)`, `EXPAND_ENTITY(slot)`, `ANSWER_AND_STOP`, `NEW_QUERY`처럼 typed legal action을 사용한다. ID는 문자열로 생성하지 않고 런타임 candidate table의 인덱스로 전달한다.

장점은 도구명뿐 아니라 유한 후보로 표현되는 인자 생성까지 제거한다는 점이다. 한계는 `P_decision`이 그대로이고 후보 밖 대상을 발명할 수 없다는 점이다. `NEW_QUERY`를 명시적으로 남겨야 기존 agent의 검색 능력을 유지한다. ANSWER는 최종 답변 생성을 트리거하며, 정답 자체를 head가 내는 것이 아니다.

### B. Cached-state joint pointer + selective refresh (**추천 연구 가설**)

질문 embedding을 최초 한 번 계산하고, 실제 확보한 증거의 embedding/공개 관측 metadata를 증분 결합해 작은 state vector를 유지한다. 후보 item embedding은 corpus 구축 때 cache한다. 작은 head가 후보와 상태를 상호작용시켜 다음 item을 바로 고른다. 대부분 iteration에서는 VLM controller prefill과 query 생성이 모두 없다. 새 증거를 실제로 읽어야 하는 경우의 reader 비용, 최초 질문 encoding, cache 구축/메모리 비용은 여전히 존재한다.

캐시가 불충분하면 `REFRESH_OR_NEW_QUERY`를 택해 full VLM으로 상태를 재검토하고 필요한 검색어를 만든다. 후보 부족과 판단 불확실성을 구분해 기록한다. 최신 embedding이 image semantics를 충분히 보존한다는 가정을 검증해야 한다. 이미지 embedding만으로 숫자 OCR·관계·비교가 해결된다고 가정하지 않는다. 주된 연구 질문은 **어떤 증거 접근 효과가 cached representation만으로 보존되며, 어떤 상태는 full multimodal 재해석이 필요한가**다. 이는 현재 교차 접근 supervision과 연결할 수 있지만 연결 자체가 유효하다는 증거는 아직 없다.

### C. Intermediate-layer exit head / speculative draft

VLM 중간 layer의 state에서 일찍 행동을 고르거나 작은 drafter를 사용한다. 중간 hidden을 hook으로 받기만 하고 full forward를 완료하면 속도 이득이 없다. 실제 layer 중단 경로와 fallback이 필요하다. 멀티모달 입력의 vision encoder 비용도 남는다. 매번 full VLM이 draft를 검증하면 동기 실행에서 `P_decision`이 없어지지 않는다. 한 GPU에서는 모델 경쟁과 불필요한 branch 실행이 이득을 상쇄할 수 있다. 현재 단계에서는 B보다 구현·검증 부담이 커서 후순위다.

## 3. B의 최소 아키텍처: 동적 후보와 종료를 같은 척도로

1. 후보집합 `C_t`는 관측 가능한 질문/현재 증거를 이용한 고정 retrieval 규칙으로 구성한다. text/image modality별 shortlist를 합치되 실험군 모두 같은 후보를 사용한다. gold support로 후보를 제한하지 않는다.
2. 초기 state `z_0 = MLP(question_embedding, budget)`; 이후 `z_t = Update(z_prev, observed_embedding, modality, retrieval_score, spent_budget)`로 갱신한다. 시작 구현은 permutation-aware/순서 metadata가 있는 pooling+MLP로 충분하다. candidate의 표현은 공개 text/image embedding과 공개 modality뿐이다.
3. `g_t = MLP(z_t)`, `k_i = W_e e_i`; `score(a,i) = MLP([g_t,k_i,g_t*k_i,modality,budget])`. 작은 hidden dimension 128, shared scoring MLP로 미지 ID·가변 후보 수를 처리한다. ID embedding을 외워서 source leakage를 만들지 않는다.
4. `score(STOP)`, `score(REFRESH)`도 같은 return target 단위로 학습한다. 모든 legal `(a,i)`와 STOP/REFRESH를 하나의 action set에서 비교한다. 먼저 독립 modality softmax 후 STOP confidence와 비교하면 후보 수와 각 분모가 달라져 비교가 깨질 수 있다.
5. outcome utility의 단위는 `answer_F1 - lambda * measured_remaining_seconds` 또는 정해진 total budget의 expected final utility다. retrieval relevance는 그 자체로 STOP utility와 비교할 수 없다. 접근효과 `N0/J`도 utility와 동치가 아니므로 별도 auxiliary로 두고 이득을 실험으로 확인한다.
6. 모든 candidate action의 counterfactual rollout은 비싸다. 학습 때 shortlist를 작게 고정하고 candidate당 rollout으로 supervision을 만든다. 수집 비용도 보고하며, 현재 v1.1의 modality Q3 label을 그대로 candidate별 target이라고 간주하지 않는다.

초기 출력은 예를 들어 `{action: READ_IMAGE, slot: 3}`이고 runtime은 `read_image(candidates[3].item_id)`를 호출한다. 후보 ID를 prompt에 나열하고 VLM이 한 토큰 라벨을 고르는 대조군도 둔다. 이는 같은 후보 제한을 받으므로 인자 생성 제거에 대한 공정한 기준선이다.

## 4. Cache / KV / query-likelihood의 구별

- **Embedding cache:** query-independent semantic vector다. 충분한 정보가 있다는 보장은 없다. 이를 이용한 value prediction은 새로 학습·검증할 가설이다.
- **LM KV cache:** 정확히 같은 prefix와 위치/attention 의미를 공유할 때 재사용할 수 있다. 문서 단독 KV를 임의의 다른 대화 위치에 끼워 넣는 것은 원래 contextual forward와 동등하지 않다. 원시 KV 자체가 relevance score나 expected answer improvement를 제공하지 않는다.
- **UPR:** `log P(question | passage)`를 계산한다. known question tokens를 teacher-force하면 autoregressive sampling 없이 token likelihood를 계산할 수 있지만, query와 문서가 결합된 contextual forward와 vocab logits가 필요하다. 현재 `logits_to_keep=1`의 마지막 state만으로 전체 question likelihood를 얻지 못한다. 문서 KV를 앞부분에 cache해도 query tokens를 그 KV에 조건화하는 비용은 남는다. 이것을 'cache vector만으로 바로 UPR 점수'라고 부르면 안 된다.
- **2-stage non-generative scoring:** 기존 embedding으로 shortlist한 뒤 작은 pairwise head로 score하면 온라인 비용이 작다. 반대로 shortlist마다 full VLM teacher-forced scoring을 하면 sampling만 없어지고 full conditional compute는 남는다. 두 방식을 별개 baseline으로 계측한다.

## 5. 기존 연구가 이미 제공하는 것 (1차 출처 확인)

| 출처 | 확인한 내용 | 본 제안에 대한 제약 |
|---|---|---|
| [ToolkenGPT, 2023](https://arxiv.org/abs/2305.11554) | frozen LM에 tool embedding/token을 학습하고 이후 arguments 생성 | tool choice readout 자체는 새롭지 않다. typed pointer로 유한 인자까지 제거하는 부분과 상태 비용 절감 효과를 구분 |
| [Pointer Networks, 2015](https://arxiv.org/abs/1506.03134) | 입력의 가변 위치 집합을 pointer로 선택 | candidate pointer 구조 자체를 신규 발명으로 주장 불가 |
| [InnerJev / General Decision Models §6, 2026-10 v1](https://arxiv.org/html/2610.03935v1) | 첫 answer 위치의 label logits를 single prefill로 읽고 reasoning distribution 증류 | non-generation first-token과 reasoning distillation은 필수 비교군. prefilling 비용은 남음 |
| [Kev, 공식 저장소](https://github.com/jaredpalmer/kev) | Qwen 기반 Jev-like decision family, typed choice/score와 calibrated output 제공 | 독립 공개 구현; 비공개 Jev 내부 구조를 밝혀낸 논문으로 취급하지 않음 |
| [UPR, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.249/) | passage에 조건화한 input-question likelihood로 rerank | likelihood는 relevance baseline이며 STOP utility/정보획득 가치와 동치가 아님 |
| [ColBERT, 2020](https://arxiv.org/abs/2004.12832) | query/document 독립 encoding과 late interaction, offline document representations | precomputed candidate interaction은 확립된 IR 설계. 단일 embedding 대비 세밀한 정보 보존 대조 가능 |
| [AutoCompressors, 2023](https://arxiv.org/abs/2305.14788) | 긴 문서를 학습된 summary vectors로 압축해 soft prompt로 사용 | 임의 KV/embedding pooling의 충분성을 보장하는 논문이 아님; 추가 학습 부담 비교 필요 |
| [CoRank, 2025 v2](https://arxiv.org/abs/2505.13757) | scientific retrieval에서 compact semantic document features로 listwise reranking | compact 후보 표현이 이미 연구됨; candidate set 확대 효과와 제어 학습 효과 분리 |
| [CALM, 2022](https://arxiv.org/abs/2207.07061) | confidence 기반 동적 layer allocation/early exit | early-exit 아이디어 자체는 기존 연구이며 실제 연산 생략과 품질 통제가 요구됨 |
| [Speculative Macro Commit, 2026-09](https://arxiv.org/abs/2609.03236) | actor/drafter와 isolated execution으로 tool-chain latency 절감 | speculation도 최신 경쟁 방향. 저지연 주장에는 serial time·추가 speculative work 모두 계측 |

논문 제목·abstract와 필요한 본문(InnerJev §6/appendix)을 확인한 범위다. 모든 논문의 재현 실험이나 신규성 전수 조사를 완료한 것은 아니다. 최근 연구가 있다는 이유만으로 ICML 2027 채택 가능성을 예측할 수 없다.

## 6. 공정한 실험과 반증

최소 비교: (i) generated tool+args, (ii) same VLM first-token action+generated query, (iii) same candidates first-token pointer, (iv) full-state learned pointer, (v) cached pointer without auxiliary, (vi) cached pointer with accepted access supervision, (vii) refresh-enabled cached pointer. 모든 군의 candidate construction, reader, corpus, budget을 고정한다. fixed/heuristic retrieve and stop도 유지한다. 각 군이 더 많은 retrieval/후보/훈련 rollout을 받으면 별도 budget-matched 비교를 한다.

**먼저 확인할 것:** generated query의 결과가 non-generative shortlist에 얼마나 포함되는지 평가한다. shortlist의 gold recall은 오프라인 진단으로만 사용하며 입력에 넣지 않는다. shortlisted oracle도 못 푸는 문항이 많으면 pointer 학습 전에 후보 설계를 수정해야 한다. list size 증가가 retrieval cost/stop bias를 바꾸는지 별도 확인한다.

보고: end-to-end F1/EM, bootstrap source-group CI, p50/p95 total latency, state-update/full-prefill/query-generation/reader/retrieval 비용, refresh 및 new-query 비율, shortlist failure, 실제 controller VLM 호출 수, 총 GPU 시간·cache 구축/메모리. head inference milliseconds만으로 speedup을 주장하지 않는다. cache warm/cold와 최초 embedding 비용을 별도로 보고한다.

**미리 정할 중단 기준:** 공통 quality floor(개발셋에서 미리 정하고 test 고정)에서 cached controller가 full-state first-token pointer의 latency–quality frontier를 개선하지 못하면 B 가설을 지지하지 않는다. new-query/refresh가 거의 모든 step에 필요하거나 candidate 제한으로 품질이 떨어지면 'generation-free agent' 주장을 포기한다. access auxiliary가 same-data/same-capacity generic value head보다 개선하지 못하면 그 supervision은 기여에서 제외한다.

추천은 B를 단독 대규모 framework로 벌리는 것이 아니라, **관측한 증거의 어느 정보가 다음 acquisition에 필요한지에 따라 cached dispatch와 VLM refresh를 구분하는 좁은 가설**을 A 및 first-token pointer로 반증하는 것이다. 현재 수집은 modality 선택/접근효과의 존재 여부를 검증하는 데 쓰고, candidate supervision·cache sufficiency는 별도 후속 pilot로 취급한다.
