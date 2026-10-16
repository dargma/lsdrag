> Latest execution checkpoint: external Qwen/D3 dev10 diagnostic completed; see `lab/reports/text_integrated_20261016/d3_qwen_dev10.md`. Learned heads are pending. dev10 is now exposed; future frozen holdout IDs are in `lab/registry/text_integrated_20261016/next_holdout_reserved.json`. Argument generation contract must be hardened on train-only examples first.

# Latest override — Qwen3-4B non-thinking reader and prompt audit (2026-10-16)

This section overrides all Gemma choices and superseded scale/evaluation details below. User now permits LLM or VLM, asks for2–4B, prompt-setting inspection, and as little thinking mode as possible. Selected `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`: official4.0B text-only causal LLM, non-thinking-only release. Source: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 . Newer Qwen3.5-4B defaults to thinking; the explicit non-thinking-only requirement takes priority over release recency. No model sweep.

Prompt audit: Gemma prepare passed enable_thinking=False; actual rendered prompt has no `<|think|>` activation. Observed thought-channel output is not evidence that thinking was enabled. Native parser leaves a spontaneous thought block in content and strict JSON parsing fails. Original task instruction also made JSON conditional on a runtime FINAL request, which is ambiguous for spontaneous native final output. Prompt v2 `prompts/text_rag_integrated_v2.txt` unconditionally requires shortest answer span JSON for every final; no analysis/thinking/extra fields. A non-thinking-only Qwen template prefix is checked before inference; no media and no thought activation/prefix allowed.

Preserve old `d3_native_20261016` first2×2 episodes as format-confounded diagnostic only. Native strict F1=0 and D3 F1≈0.214 do NOT establish decision superiority. Original Gemma conditions shared a backbone but had different final realization/serialization. Do not reuse its labels/features in Qwen experiments.

The external D3 benchmark now isolates selection: Qwen first-token execution-choice scoring vs D3 execution-choice scoring, identical candidate builder/schema and Qwen argument/final generation, text embedding/corpus/budgets. This is a zero-shot baseline, not native A-RAG and not a trained action head. D3 still adds a separate9B decision backbone and its full costs are counted. Runtime/script: `scripts/benchmark_d3_qwen_text_rag.py`; non-thinking causal adapter `src/lsdrag/text_only_reader.py`. Qwen CausalLM uses its own tokenizer/template; Gemma transport/parser is not reused.

Frozen original dev12 first2 were inspected for format debugging. Exclude BOTH of them from new comparison independent of outcome; use fixed remaining dev10. Functional preflight uses train first2, not dev. No additional dev replacements or tuning. Future primary native/head-only/integrated/D3 conditions use the same remaining10 (max40 eval episodes); native Qwen tool parser and learned-head implementation remain pending. Teacher plan train32→48 cap unchanged; fresh Qwen candidate-aware pre-output features/labels only. Seed20261016 and total worker60min ceiling unchanged.

New output namespaces: `/content/local_fast/lsdrag-text-v1/results/d3_qwen_preflight_20261016` (train2) and `/content/local_fast/lsdrag-text-v1/results/d3_qwen_dev10_20261016` (dev10). Plan/model/prompt changes are recorded before new runs. Never resume old outputs under changed source. Next: complete Qwen file verification, train2 functional preflight, then unchanged dev10 comparison.

---

# Text-only RAG: 행동 선택과 인자 생성의 통합 효율화

상태: 사용자 최신 지시에 따른 새 소규모 계획. 계획·instruction 초안만 저장했으며 구현, 데이터 준비, teacher 수집, head 학습 및 평가는 아직 시작하지 않았다. 이전 멀티모달 v13/v14 자동 실행 명령을 사용하지 않는다.

## 최신 사용자 지시와 우선순위

2026-10-16: (1) 행동 head와 인자 생성 최소화를 동시에 평가, (2) GPU 제약 때문에 필요한 만큼만 실행, (3) 이미지 입력·이미지 도구를 제외하고 text-only 벤치마크로 교체, (4) 최신 중간 크기 Gemma VLM은 유지하되 텍스트만 입력, (5) instruction도 변경, (6) 변경사항을 영속 기록해 향후 혼동을 방지.

이 문서는 과거 full-tool/이미지 전달/6-action/60→80→100/train100-dev20 지침보다 새 실험에서 우선한다. 과거 코드·실험·데이터는 원래 identity로 보존한다. 이미지 기능을 기존 구현에서 삭제하는 작업은 아니다. 새 전용 text-only profile을 구현해야 한다. World model/surprise는 계속 제외한다.

## 선택한 모델과 벤치마크

- Reader/teacher/backbone/새 인자 생성/최종 답변: `google/gemma-4-E4B-it`, revision `ee0ef6023621cff504d758262d4e04895a5af4a2`. 2026-10-16 Hub current revision과 일치함을 확인했다. 새 E4B snapshot은 pinned revision으로 다운로드/검증한다. 12B 복원 검증을 E4B 검증으로 간주하지 않는다. E4B는 약4.5B effective, embedding 포함 약8B이며 BF16 weight 파일은15,992,595,884bytes다. BF16 frozen backbone 하나를 공유하고 텍스트만 전달한다.
- 데이터: **MuSiQue-Ans v1.0의 2-hop 부분집합**. 공식 train에서 teacher, 공식 dev에서 평가를 고정 선택한다. MuSiQue-Full의 unanswerable 대비 실험은 이번 범위에서 제외한다.
- 이유: 연결된 다단계 추론과 새 검색어/근거 ID 선택이 필요해 인자 재사용의 이득과 한계를 시험하기 적합하다. HotpotQA는 이번에 추가 실행하지 않는다.
- 문항별 공식 제공 paragraph candidate 집합만 검색하는 bounded-corpus 설정. 전체 문단을 reader prompt에 미리 넣지 않고 검색→snippet→read로 점진 공개한다. Full-Wikipedia/open-domain RAG 결과로 부르지 않는다. 문단 수·길이는 실제 파일을 받은 뒤 기록하며 임의 고정 개수로 가정하지 않는다.
- 고정 seed `20261016`으로 2-hop/train 최대48, dev12의 ID 목록을 성능 관측 전에 저장한다. 동일 question/decomposition family와 중복 질문은 train/dev 교차를 검사한다. 원본 split을 유지하며 unseen-document 또는 사전학습 비노출은 주장하지 않는다.
- 원문 paragraph만 corpus에 둔다. answer/aliases/support flags/decomposition answers는 private supervision/scoring 영역에만 둔다. 이 필드로 후보 구성·검색·runtime prompt를 만들지 않는다.
- MuSiQue 공식 answer EM/F1(aliases 포함) scorer를 pinned source 그대로 사용한다. predicted support IDs 기반 support F1과 retrieved/read support recall은 각각 별도 보고한다. 기존 MMQA list scorer는 사용하지 않는다.
- 공개 원본은 공식 배포 링크에서 local-fast로 다운로드하고 SHA256을 기록한다. 공식 공개 Google Drive 배포 링크는 사용자의 백업 Drive와 구분한다.

공식 근거:
- Google 모델 카드: https://ai.google.dev/gemma/docs/core/model_card_4
- 모델: https://huggingface.co/google/gemma-4-E4B-it
- MuSiQue 논문: https://arxiv.org/abs/2108.00573
- 데이터/채점기: https://github.com/StonyBrookNLP/musique
- 확인한 MuSiQue HEAD: `922ac98f19a201998dbdae6d7f2887a5258dbdeb` (코드/공식 다운로드 스크립트 pin; 데이터 bytes hash는 다운로드 후 별도 기록).

## 도구·instruction·실행 계약

도구 순서: `keyword_search`, `semantic_search`, `read_chunk`, `FINAL`의 **4행동**. 이미지 검색/읽기/자동 이미지 전달은 없다. text-only semantic index만 사용하며 기존 Gemma embedding snapshot을 텍스트 임베딩 용도로만 사용한다. 검색 인덱스는 새 데이터에 대해 새로 만든다. 기존 MMQA corpus/index/labels는 재사용하지 않는다.

새 instruction 초안: `prompts/text_rag_integrated_v1.txt`. 이미지·표 캡션·figure·visual verification 관련 지시를 넣지 않는다. 최종 출력은 `answer` 문자열과 `support_ids` 배열. 이 답변 형식은 새 parser/scorer adapter를 함께 구현해야 하며 현재 reader가 이미 지원한다고 주장하지 않는다.

구현 전 prompt·JSON Schema·4행동 순서·head dimensions·arg slots·handler·최종 parser·scorer를 하나의 새 계약으로 동결한다. text-only profile에는 이미지 tool이 schema에 없어야 하고 image content/pixel inputs는 명시적으로 거부해야 한다. 기존 멀티모달 프로필은 보존한다.

## 통합 head: 별도 인자 선택용 VLM 호출 없음

하나의 candidate-aware student prefill로 얻은 동일 hidden state에서 (a) 4-way action linear head, (b) 도구별 5-way argument-slot linear head(관측 후보 최대4 + `GENERATE_ARGS`)를 계산한다. 재사용 후보 선택 시 Python이 schema-valid 인자를 구성해 기존 handler를 바로 호출한다. 별도의 후보 선택 prefill/LLM decode를 추가하지 않는다. FINAL은 최종 답변 생성으로 간다.

- `read_chunk`: 실제 tool 결과에 노출된 ID/인접 ID만 사용. 최근 순서 보존 1–8개 묶음과 단일 ID 후보를 결정적 순서로 최대4개 제시한다. 다른 조합은 생성 경로에서 표현 가능하다.
- `semantic_search`: 원 질문 그대로 또는 실제 이전 query/인자 전체의 재사용. 새로운 multi-hop query는 생성한다.
- `keyword_search`: 실제 이전 keywords/인자 배열만 재사용. 첫 호출이나 새 keywords는 생성한다. 원 질문 전체를 무조건 keyword로 바꾸지 않는다.
- 후보는 정해진 response 필드와 실제 과거 call에서만 구성한다. 자유 텍스트에서 ID를 임의 추출하지 않고 corpus 전체/gold/미래 teacher output을 참조하지 않는다. 문항마다 초기화한다.
- 검색 기본 top_k=5; 이전의 명시적 top_k도 보존한다. 빈 후보 slot을 mask하고 후보가 없으면 arg head 선택 없이 `GENERATE_ARGS`. 새 범위/다중 ID/다른 top_k는 원 스키마 생성 경로로 표현 가능하다.
- candidate list·slot 순서·기원·해시와 selected slot·dispatch args를 매 결정 저장한다. 후보 텍스트를 넣는 CPU 처리와 prompt 비용도 전체 비용에 포함한다.
- 학습 목표: action CE + non-FINAL에 대한 conditional arg-slot CE(계수1). native teacher의 schema-valid 인자가 현재 후보와 기본값 정규화 후 정확히 일치하면 해당 slot, 아니면 `GENERATE_ARGS`. 복수 일치 시 첫 slot. 의미적으로 비슷하다는 이유로 재사용 label을 만들지 않는다.
- teacher action/args가 나오기 **전** student feature/candidate를 확보한다. valid earlier states from unsuccessful episodes도 유지하며 forced FINAL/invalid native targets는 학습 label에서 제외하고 분모·실패를 보고한다. native teacher 인자는 teacher rollout만 실행한다.
- 이 작은 linear slot head가 동적 후보 의미를 충분히 읽는지는 검증 대상이다. cached controller/early exit/RL/LoRA/새 query writer를 함께 넣지 않는다.

## 필요한 만큼의 실행 규모

| 단계 | 규모 | 진행 조건 |
|---|---:|---|
| CPU+synthetic smoke | 실제3개 tool, FINAL, reuse/generate 양 경로 | schema/dispatch/상태 격리/비용 계측 검사; 실험 품질 증거 아님 |
| fresh teacher 사전 점검 | train4 | 최종32에 포함. 형식·readout·infra 실패는 자동 확장 중단 후 review |
| fresh teacher 수집 | train32 | 미리 고정한 순서. coverage 부족할 때만 최대48 |
| head 학습 | seed20261016, 100epochs, CPU | 아래 coverage gate 통과; backbone gradient 없음 |
| paired 평가 | untouched dev12 ×3조건 =36 episodes | 모든 조건/문항/실패 공개. best seed/질문 선택 없음 |

Coverage: 자연 teacher state가 있는 distinct train qid>=30, 4행동 각각>=5 distinct qid. 인자 supervision은 reuse와 GENERATE_ARGS가 각각>=5 distinct qid에서 있어야 한다. 48에서도 실패하면 full-action 또는 argument branch가 학습 가능한지 확인되지 않았다고 NoGo를 기록하고 학습·평가 완료로 표현하지 않는다. 드문 행동을 강제로 생성하지 않는다. 기존60/80/100과3-seed GPU 평가는 이번 작은 파일럿에 적용하지 않는다.

평가 조건:
1. **Native reference**: 같은E4B·3도구·답변 계약의 native single-call controller. 정확한 원 논문 A-RAG 재현으로 부르지 않는다.
2. **Head + always-generate**: 통합 모델의 동일 action head, 같은 candidate-aware 입력을 쓰되 인자는 매번 생성한다.
3. **Head + integrated args**: 같은 action head와 arg-slot head로 재사용/필요한 인자 생성.

2↔3이 인자 경로를 분리하는 주 비교다. 같은 초기 모델/입력을 쓰지만 이후 rollout은 분기할 수 있다. 2는 candidate prompt overhead를 공유하는 controlled ablation이며, 최소 prompt로 최적화한 기존 action-only 모델이라고 주장하지 않는다. 전체 시스템 비용은 native reference와도 비교한다. Native reference와 head의 instruction/selection 구조 차이를 명시한다.

GPU 상한: loadedE4B 하나, batch1, 조건은 순차 실행. text embedding은 한 번 precompute/cache하며 query encoder는 작은 고정 모델을 사용한다. context16K, 최대8 tool calls, 각 생성512 tokens, greedy, thinking off를 모든 조건에 공통 적용한다. 과거32K/1024와 다른 새 계약이다. 문단이 context를 초과하면 동일 규칙으로 budget FINAL 처리하고 숨겨진 truncation을 하지 않는다.

실험 worker 누적 wall budget **60분**(모델 로딩·teacher의 student probe·retrieval·evaluation 포함)을 초기 자원 상한으로 둔다. 이는 예상 소요 시간/보장된 완료 시간이 아니다. launch 전 32GiB free VRAM 확인; 부족하면 기다리고 다른 작업을 종료하지 않는다. deadline에서 새 model call을 시작하지 않고 완료된 artifact와 pending 목록을 저장한다. 시간 상한 도달은 결과 NoGo와 구분되는 budget-limited/incomplete다. 반복 retry·모델 sweep·벤치마크 추가·80/100 확장은 자동으로 하지 않는다.

## 판단 지표와 다음 단계

주 결과: official answer F1/EM, support read/retrieved recall, format/dispatch 실패, 생성한 인자 tokens/calls, reuse/GENERATE_ARGS 비율, total prompt/completion tokens, 모델 prefill 횟수와 구성요소별 비용. final answer tokens와 teacher/probe collection cost를 deployment cost와 구분한다. 조기 종료/실패 때문에 토큰이 줄어든 경우를 효율화 성공으로 세지 않는다.

dev12에서는 문항별 paired 차이와 bootstrap 불확실성을 탐색적으로 보고한다. 1개 seed와12문항은 안정적인 +5% 품질/+5% 속도 입증 규모가 아니다. 기존 +5%/+5%는 장기 목표로 유지한다. Shared GPU 시간은 관측값만 제시하고 comparative speed gate는 통과했다고 주장하지 않는다. 별도 간섭 통제 timing과 다중 seed/큰 평가셋은 이번 파일럿의 신호를 본 다음 단계다.

새 작업 namespace: `text_integrated_20261016`; 예정 runtime `/content/local_fast/lsdrag-text-v1`, 결과 `/content/local_fast/lsdrag-text-v1/results/text_integrated_20261016`, 백업 `/content/drive/MyDrive/lsdrag-backups/20261016/text_integrated_20261016`. 아직 이 실험의 launch command/PID는 없다. 다음 작업은 text-only profile/parser/통합 head/수집·평가 runner 구현과 공식 데이터 bytes 검증이다.

## 추가 사용자 지시: D3-Flash9B 외부 비교 (2026-10-16)

사용자가 Decision3.0 D3-Flash9B를 찾아 benchmarking하도록 명시적으로 요청했다. 공식 모델은 `vllm-sr/d3-flash@8b35c853b90e777595ec36bacae820fe08ea077a`, Qwen3.5 기반8.39B(vision 포함)이며 arbitrary text를 생성하지 않고 supplied choices의 확률을 읽는다. 모델 카드: https://huggingface.co/vllm-sr/d3-flash .

주 실험 Gemma backbone을 D3로 교체하지 않는다. D3는 외부 pretrained controller baseline으로 추가하고, 새 인자·답변은 동일 Gemma E4B가 생성한다. 통합 도구+인자 실행 후보를 한 choice 질문으로 주어 선택하되 후보 밖은 도구별 GENERATE_ARGS를 제공한다. 여러 질문 API가 하나의 forward를 공유한다고 가정하지 않는다(공식 설명은 질문마다 forward). No image inputs.

먼저 새 fixed dev12에서 native Gemma와 D3+Gemma의 paired 소규모 비교를 준비한다. 향후 learned-head 조건도 같은 목록에 평가한다. 최종 조건 수는 기존3+외부D3=4, 최대48 eval episodes다. 이번60분 worker budget은 조건 추가로 늘리지 않는다. 자원·시간이 부족하면 완료 subset과 pending을 명시한다. 전체 Jev Index140k requests를 실행하는 요청으로 해석하지 않는다.

모델/환경은 isolate한다. D3가 권장하는 transformers5.17.0/custom runtime을 별도 환경에 두고 공식 코드/manifest를 검토·검증한다. AutoModel text smoke와 소규모 decision/RAG benchmark를 구분하며 vendor Jev/MI325X 성능 수치를 우리 GPU 결과로 제시하지 않는다. 서로 다른 backbone과 추가9B controller 비용을 모두 보고하고 전체비용상 불리할 가능성도 평가한다. E4B와 D3를 동시에 load하려면 각각 단독 실행보다 큰 VRAM이 필요하므로 실행 직전 둘의 실제 가용량을 확인한다.
