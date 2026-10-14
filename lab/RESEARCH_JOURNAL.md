# 실험 진행 기록

## 2026-10-13 계획 고정과 사전 확인 착수

사용자 최신 지시를 실행 지시서와 결정 기록에 반영했다. 최종 평가 범위를 이미지 관련 200개로 줄이고, V2 인터페이스 재구성과 Reader 축소를 허용하며 추가 승인 절차를 제거했다. A-RAG 기본 소스에 rerank가 없음을 확인했다. 로컬 작업 체크아웃과 A-RAG/M3DocRAG 소스를 확보했다. GPU는 96GB급이며 로컬 가용 디스크가 약 39GiB여서 다운로드 전에 자원 계획을 검증한다. 모델 실제 실행, 데이터 확보 및 평가는 아직 완료되지 않았다.

## 2026-10-13 환경과 데이터 사전 확인

두 임베더의 GPU 실행, MinerU의 실제 PDF 1페이지 파싱·이미지 저장, Gemma 4 E4B의 도구 호출 및 user-role 이미지 해석을 확인했다. 원본 A-RAG 및 PDF 유틸리티 테스트 13개와 질문 선정 테스트 2개가 통과했다. 라이브러리 충돌 때문에 임베딩·Reader·파서를 분리했다. Reader는 저장 공간 제약으로 사용자 허용 범위 내 E4B로 축소했다.

MMQA dev 2,441개 중 이미지 근거 후보 940개를 확인하고 개발 50개와 본 평가 후보 200개를 근거 URL까지 분리했다. 개발 코퍼스 183개 렌더링 결과 정상 174개, 이미지 검토 필요 7개, HTTP 404 실패 2개다. 정답 매핑용 원본 이미지 192개도 확보했다. 전체 메타데이터 기준 코퍼스는 3,366개 ID이며 논문 수치와 차이를 기록했다. 상세 근거는 `lab/reports/PREFLIGHT.md`에 저장했다. 현재 표준 tier의 3페이지 파싱을 점검 중이며, 검색 인덱스·에이전트 구현·본 평가는 다음 단계다.

표준 tier로 3페이지 재검증을 통과했다(분석 8.8초, 모델 준비 포함 18.0초). 개발 코퍼스의 첫 10개 문서 파싱을 시작했다. 사전 확인 데이터 835개 파일·360,673,466바이트를 Drive에 백그라운드 복사하고 해시 일치를 확인했으며 실패는 없었다. 가중치는 고정 revision으로 다시 받을 수 있으므로 이번 데이터 백업에서 제외했다.

## 2026-10-13 — IR and retrieval implementation

Standard parser pilot: 10/10 documents, 212 pages, 109.59 document-processing seconds. Initial IR has 2,958 chunks and 196 figures; these counts precede the footnote-description correction. MinerU stores some real figure descriptions as footnotes, now retained alongside captions. Implemented verified saved-asset resolution, document URL aliases, page-preserving chunks, table header repetition, pinned full-dimensional index construction, rank-only fusion, and V2 image search. Six tests pass. Agent loop, actual indexes, visual mapping and benchmark results are still pending. Development parsing will resume over the collected PDFs; no final evaluation is starting.

## 2026-10-13 — Agent transport and inference budget

Implemented native tool-call iteration with text-only search responses, read-triggered image delivery in subsequent user messages, separate opened/accepted figure traces, forced final answers, token and latency accounting, and immutable run manifests. B0/V1/C1 share the exact prompt and schemas; V2 adds explicit visual search and reading. All arms use 60 text candidates, selected from top 180 sentence scores, then return 5 by default; V1/V2 additionally fuse 60 image candidates by RRF k=60. This common candidate pool differs from original top_k*3 filtering to keep the text channel fixed for the fusion comparison. No reranker. Original upstream remains separately pinned for reproducibility. Eleven tests pass. Gemma E4B successfully recognized a synthetic red circle with max_soft_tokens=1120 (Reader response saved); no benchmark conclusion follows from this check.

## 2026-10-13 — B0 end-to-end runtime smoke

Qwen index: 12,362 sentences, 2,560 dimensions; Gemma image index: 196 figures, 768 dimensions. One development question with both supporting documents in the 10-document pilot completed B0 in 7.25 seconds, four Reader calls, one accepted image, and valid JSON output. It did not answer the stripe-count question correctly; this proves transport and tracing, not retrieval quality. It inspected a Panthers logo instead of resolving the stadium sponsor logo. Full development evaluation and evidence availability audit remain pending.

## 2026-10-13 — User scope update from side conversation

User explicitly requested recording and continuing with M3DocVQA 100 image-related questions and supplementary Mem-Gallery 60 questions (30 VS + 30 VR), with an oracle-evidence Reader diagnostic. Updated plan, decisions and state only. Existing processes, code, manifests and results were left intact. The main execution must regenerate/freeze the 100-question manifest before final evaluation, then perform the separately reported Mem-Gallery comparison. No new evaluations were launched by this documentation update.

## Latest user scope revision — 50 questions per benchmark

User reduced both final evaluations to 50 questions: M3DocVQA image-related 50 and Mem-Gallery VS 25 + VR 25. Updated plan, decisions and approved scope; earlier 100/60 figures are historical. Development remains separate. Final input manifests must be frozen to the new scope before evaluation. Active processes, code, data and results were not modified.

## 2026-10-13 — Scope synchronization, scoring and visual QC

Detected and preserved the user-approved side-conversation plan updates: final M3DocVQA 50 and Mem-Gallery 50 (25 VS + 25 VR), with oracle-evidence Reader diagnostic. Selected the M3DocVQA 50 from the original 200 candidates by seeded proportional strata; development inputs and original candidate provenance remain unchanged. No final-question model outcomes informed selection. Mem-Gallery official repo cloned and 20 dialog JSONs downloaded at HF revision af912daba984e896e253016b7c7e334ef92c2a6f; image downloads and schema-specific pipeline remain pending.

Official MMQA scorer verified numeric equivalence and multi-answer alignment. B0 pilot scored zero EM/F1/contain/LLM accuracy and 0.5 document-read recall, consistent with inspecting only the team logo. V2 pilot exercised image_search and figure_read but still stopped on the wrong entity. Added one explicit final verification turn within the shared loop budget, and repeated the original question in image-delivery messages for all arms. This development adjustment is unvalidated on the full development set and is not evidence of an accuracy gain. Mechanical JSON recovery now handles prose before a JSON object while recording the format violation. Fifteen tests pass.

Image alignment proposals use within-document pHash and SIFT/RANSAC; proposals are not gold labels. Visual inspection of five pilot originals/candidate sheets confirmed three matches (Mini, Make It Rain, Bank of America Stadium), two unresolved (KK Krka logo changed; Tampa montage not matched among top candidates). The stadium logo is present inside a larger parsed infobox image. Store these findings separately from agent inputs.

## 2026-10-13 — Development parsing complete

All 181 collected PDFs parsed successfully (2,718 physical pages); initial collection still has two HTTP404 failures and seven PDFs flagged for incomplete article-image loading. URL deduplication yields 180 physical retrieval documents / 194 MMQA aliases, 2,693 indexed pages, 37,232 chunks and 1,510 visual regions. Of the regions, 1,380 lack a MinerU caption field but only 671 lack both caption and original footnote description. Started full development Qwen text indexing (160,271 sentences), Gemma image indexing and image-alignment proposals. Stopped the completed-stage parser server to release GPU memory. Reader server remains on port 8102. Gemma text index and full development evaluations are pending.

## 2026-10-13 — Mem-Gallery inputs and shared GPU recovery

Selected 50 final VS/VR questions (25 each; 13 with question images) and 10 development questions from two disjoint scenarios. Downloaded 990 actual referenced images; built 3,759 conversation chunks and 973 figures across 19 scenarios. Added strict scenario scope before ranking cutoff, cross-scenario read rejection, question-image transport and distinct evidence counts. Imported pinned official Mem-Gallery evaluator and verified EM/F1 behavior. Twenty tests pass.

Gemma text indexing initially failed at model transfer with CUDA out-of-memory while unrelated GPU workloads grew. Preserved failure log, stopped idle Reader, and retried with batch 16 successfully: 160,271 × 768 text vectors. Gemma image index completed (1,510 figures). Qwen indexing remains live. Restarting Reader to run C1 development while Qwen finishes; development timing under GPU contention is diagnostic and will not be used as final cost comparison.

## 2026-10-13 — Development parsing/indexing milestone

All three development indexes complete: Qwen 160,271 × 2,560 text vectors, Gemma 160,271 × 768 text vectors, Gemma 1,510 × 768 image vectors. C1 completed all 50 development questions without infrastructure failures; answer/evidence scoring is pending. Reviewed 53 original-image alignment sheets: 34 accepted matches, 19 unresolved/partial/color-variant cases. The blue versus red Minnesota Fighting Saints logo demonstrates why automatic geometric matches are insufficient. Added metrics that separate candidate appearance, opening, actual delivery and image-evidence page recall; unverified mappings remain null, not failures. Twenty-two tests pass.

Background copy verified all 4,134 parsing/IR/manifest/alignment files (1,286,081,444 bytes), zero failures. This backup does not yet include the completed vector indexes, Mem-Gallery data or active evaluation outputs. Native text-only A-RAG dispatcher and original example settings added; unchanged upstream CLI dispatch verified via --help, not a reproduction of published scores.

## 2026-10-13 — First complete development scores and oracle diagnostic

C1 development: 50 questions, EM .02, F1 .044, contain .10, A-RAG judge .02; complete image mappings for 33 questions, with 2/33 gold-image delivery. Document recall .5367. These are development diagnostics, not final benchmark findings. B0 development is running. Completed 10 Mem-Gallery development oracle requests using annotated clue turns/images only, never gold answer text; official scoring is running. All selected oracle examples fit the common eight-image budget (development maximum four, final maximum six).

Verified Mem-Gallery judge implementation mismatch: five-level prompt outputs are collapsed by official code to 0/.5/1. Retain both official and raw-rubric scores. Added scenario-level paired bootstrap sensitivity analysis. Twenty-three tests pass. Updated README to current 50+50 scope and added native A-RAG reproduction entrypoint documentation.

## 2026-10-13 — Reader diagnostic and evidence annotation audit

Mem-Gallery development oracle (10, clue-only version): EM .60, F1 .72045, official judge .75, raw five-level judge .70. This establishes useful Reader ability with supplied evidence, not a cross-benchmark comparison against M3DocVQA. One development VS question has a gold answer image outside its annotated clue turns, producing .95 image-delivery recall despite all clues being supplied. Audited all selected VS questions: one such inconsistency in development, zero in final 50. Preserve the question; update the separate oracle path to include the original turn of every explicit target image ID, without putting an answer string into the prompt. Retain the prior diagnostic unchanged and rerun under the completed evidence policy.

### Development results and resource-limited final corpus
- M3 B0 development: 50/50 completed, EM 0.02, F1 0.0458, A-RAG judge 0.02; complete verified figure mappings for 33 questions, delivered-gold-figure recall 2/33. These are development results only.
- Mem-Gallery complete-evidence oracle development: 10/10 completed, EM 0.60, F1 0.72545, official parsed judge 0.80 (raw rubric 0.75), evidence-image delivery 1.0. Clue-only diagnostic preserved separately.
- Froze 288-URL final M3 retrieval corpus (83 support + 205 distractor URLs), started 108 new PDF downloads. Full-corpus index estimate exceeds local free disk. Composite index preserves original float32 vectors and global ranking without duplicate disk storage; five targeted integrity/retrieval tests pass.

### V1 development evaluation complete
- 50/50 questions: EM 0.04, F1 0.0948, A-RAG judge 0.08. In 33 questions with complete reviewed visual mapping, candidate recall 25/33 and delivered recall 11/33 (B0: 3/33 and 2/33). This supports testing downstream agent/Reader bottlenecks; it is not a held-out conclusion.
- V2 development started with the documented image-search prompt and one verification turn within the same 15-loop budget.
- Final PDF expansion attempted 108 URLs; initial HTTP 429 and missing image loads were retried serially before parsing. Collecting current source evidence does not imply that original gold figures survived.
- Added paired bootstrap, exact McNemar, scenario-cluster bootstrap and per-question gain/loss exports. Failure buckets are evidence-delivery observations; metric disagreement requires manual review before calling a judge error.

### Final document collection audit
- Collected and hash-verified all 288 frozen URLs (4,093 pages): 280 image-load-clean, 8 flagged (7 reused development distractors and 1 new support document). None of the final 50 questions has a missing PDF; one has an image-load-incomplete support document.
- The remaining new missing asset is the cathedral image on N'Djamena, which returned HTTP 403 both with and without its tracking query. This is recorded, not silently substituted. Gold-image visual matching remains pending.
- All 108 new PDFs (1,400 pages) entered standard-tier parsing. Mem-Gallery B0 development is running alongside M3 V2 development; development timings are affected by concurrent work and must not be presented as isolated serving benchmarks.

### Evidence-stage and subgroup reporting
- Mem-Gallery development B0 completed 10/10: EM 0.30, F1 0.435, official parsed judge 0.60, image-delivery recall 0.5333. V1 development inference also completed 10/10; scoring follows.
- Added document candidate recall separately from read recall, tool counts, official question-type and reviewed original-description subgroup summaries. Search/evidence text volume uses tiktoken's gpt-4o encoding; Reader API prompt/completion counts are actual model usage and are explicitly distinguished.
- Before any final model predictions, annotated the 50 final M3 question texts with a primary visual cue category. All 50 use a visual property; these categories are exploratory manual labels, not independently adjudicated. No answers or model outcomes were used.
- Four metric tests pass, including a candidate found but unread case.

### Native compatibility smoke and final alignment review
- Ran unchanged upstream A-RAG build_index.py and batch_runner.py through the native dispatcher with the pinned local Qwen4B embedder on CPU and the common Reader. A two-chunk/four-sentence synthetic index was built at 2,560 dimensions; keyword search and read_chunk returned the correct fact in three loops. All three original tools were registered. This is functional compatibility, not reproduction of the paper's 0.6B benchmark scores.
- Reviewed 39 final original-image alignment proposals against completed new document parses: 25 matched originals, 14 unresolved or visibly changed variants. Original photographs inside larger infobox crops remain valid matches; differing posters, collage content and logo variants are not silently treated as exact originals. Full final mapping is pending remaining parses.
- Corrected only the alignment/QC script's transparent-image visualization to composite onto white; discarding alpha hid black lettering. Retrieval images and agent behavior are unchanged. Previous partial sheets and their checksums remain preserved.
- Mem-Gallery V1 development: EM 0.40, F1 0.5171, official judge 0.55, delivery recall 0.5333. V2 development inference finished 10/10 and is being scored.

### 2026-10-13 — Forced-session recovery started
- Fast-forwarded Drive checkout by three commits to 7aaa4f0 and cloned a local working checkout. All four active model snapshots downloaded at pinned revisions.
- Restoring preflight and development parsing/IR backups with SHA256 verification. Unbacked indexes, raw prediction traces, Mem-Gallery data, and final additional PDFs/parses must be regenerated; historical aggregate reports remain preserved.
- Runtime tests: 24 passed. Recreating isolated Reader, embedding, and parser environments; actual model checks and experiment restart remain in progress.

### Recovery runtime verified and experiments resumed
- Verified 4,969 restored backup files (1,646,754,910 bytes). All four model snapshots and upstream repositories are restored at pinned revisions. Reader tool/image smoke and real one-page standard parser smoke pass.
- Mem-Gallery: 990 images; both splits and corpus hashes exactly match original. Both indexes rebuilt, V2 development inference is running in a separate recovery namespace.
- Final 288-URL selection hashes match original. All 108 lost additional PDFs were recollected; their live-source bytes are new artifacts, not claimed to reproduce lost PDF hashes. Standard parsing resumed.
- M3 Qwen index rebuild remains active. Continuation waits for it, builds Gemma indexes, backs up indexes/data, then runs and scores development V2. Separate completion workers score and back up Mem-Gallery V2 and final document parses. Concurrent development timing is diagnostic only.

### Interim report focused on the embedder replacement question
User clarified the primary question: what changes when replacing Qwen with Google multimodal embeddings? Wrote a concise interim report separating C1 text replacement from V1 image-channel addition, and explicitly marked Google-text-plus-image full replacement as unmeasured. No new experimental arm was launched. Added the completed recovery Mem-Gallery V2 summary; M3 V2 remains in progress.

### V2 development complete and illustrated report revised
- M3 V2 recovery: all 50 scored; EM .10, F1 .1366, LLM accuracy .14. On 33 fully mapped questions, figure candidate recall .9091 and delivery recall .6364. This includes tool/prompt changes, not an isolated embedding replacement.
- All 108 additional PDFs parsed; recovery-final backup verifies 2,715 files. Development indexes and resumed results backed up.
- Reorganized INTERIM_REPORT.md into the seven requested sections with a red bold numeric SVG and four illustrated cases, two per benchmark. Included helpful visual-search examples, a fine-detail counting failure, and a date-constraint failure; exported actual trace excerpts and image provenance.

### User revision: three isolated embedder conditions, eight iterations
Replace mixed Qwen/Google V1/V2 comparisons with B0 Qwen-text, C1 Google-text, G1 Google-text+image. Same VLM/prompt/tools and caption-linked image delivery in every arm. Eight total Reader calls including finalization. Mem-Gallery input is text-only for all arms, source questions/gold unchanged; no oracle runs. Remeasure existing development 50+10 questions. Examples must be paired Qwen failures and G1 successes, conditional on overall evidence of improvement. Previous report archived.

### Correct max-iteration semantics to match upstream A-RAG
User clarified that only A-RAG max iteration is eight, with no independent Reader VLM call cap. Verified pinned upstream agent/base.py: eight tool-enabled iterations, then separate forced final answer if needed. Stopped and invalidated partial three_arm_iter8_textquery runs. New three_arm_arag8_textquery runs use the corrected semantics and log agent loops separately from reader_call_count.

### Corrected three-arm comparison complete (2026-10-13)
- Completed and scored M3 50 + Mem-Gallery 10 questions per arm, once each, with eight A-RAG iterations and no independent Reader-call cap. All six runs backed up. Identical per-benchmark question/corpus/prompt/config manifests verified.
- Qwen / Google text / Google multimodal EM: M3 0/2/4%; Mem-Gallery 20/20/30%. Image candidate recall: M3 9.1/9.1/81.8%; Mem-Gallery 53.3/48.3/81.7%. Delivery recall: M3 6.1/6.1/27.3%; Mem-Gallery 43.3/48.3/78.3%.
- Updated seven-section report with recall denominators and paired uncertainty. Overall answer improvement remains uncertain; no success-only illustrated showcase or repeated question solves.
- Stopped our Reader GPU server at user request. Subsequent expansion to at least 200 questions is authorized only after 20 continuous GPU-idle minutes; no expansion launched by this report update.

### Additional 50 per benchmark and self-contained qualitative cases (2026-10-13)
- User reported GPU idle and requested 50 new questions per benchmark after updating report metrics. Used the two previously frozen 50-question selections, with zero question-ID overlap with development. No completed question was solved again. All 300 new predictions successful, scored and backed up. Total valid comparisons: 100 M3DocVQA + 60 Mem-Gallery questions, three arms each (480 predictions).
- Built only 108-document addition index shards, reused all 180-document development shards, composed immutable 288-document indexes. Mem-Gallery indexes reused. Prepared IR/index backup verified 21 files / 1,241,041,593 bytes. All 55 new gold-image proposals visually reviewed independently of model outcomes: 41 originals accepted, 37 questions fully mapped.
- New B0/C1/G1 EM: M3DocVQA 2/2/8%; Mem-Gallery 28/26/32%. Image delivery recall: M3DocVQA 8.1/16.2/32.4% (37 mapped); Mem-Gallery 52.1/50.9/52.2% (50). The large Mem-Gallery development recall gain did not persist. Paired EM intervals include zero.
- Removed answer-judge and candidate-recall columns from main report, defined cumulative Reader image-delivery recall, used full benchmark names, added four paired EM gains. Each case states original/supplied question-image status and caption/footnote presence. The Birth of Venus caption already names the answer; described as retrieval of better dialogue/caption evidence, not proof of necessary visual reasoning.
- Packaged full recorded B0/C1/G1 predictions, source question/gold, referenced source dialogue sessions or parsed PDF pages, candidate/read chunks and images, caption inventory and SHA256 files for four cases. Local asset paths are portable. Recorded predictions copied byte-for-byte; no new Reader calls for report authoring.
- Old development G1 had gold-image candidates in 27/33 questions but delivery in 9/33: 18 did not read the image chunk. Proposed automatic image reading as future common-policy experiment; current additional runs retain original policy.
- Tests: 28 passed. Report/asset links, bundle file hashes, four 0→1 EM pairs, and question text-only policy verified. Stopped our Reader GPU server after scoring.

### Correct search-to-image transport and document AI implementation handoff
- User clarified that search-hit image evidence should reach Reader VLM, and objected to ambiguous caption statements. Recorded comparisons actually used agent-selected read_chunk (`agent-read`); finding an image did not guarantee it was read. Existing results and raw predictions remain unchanged and are now explicitly labeled with that measured policy.
- Implemented `search-linked` as AgentConfig/run_experiment default: keyword/semantic returned image-linked chunks are automatically read within the common image budget; full evidence/status is added to search-result rows, actual image blocks follow before the next Reader call. No extra agent iteration or independent Reader call is added. Search ranks/list shape remain intact, dedup and budget status are audited in evidence_events, success delivery remains post-response.
- Same new policy/tool descriptions/search_linked prompt for B0/C1/G1. Historical frozen comparison coordinator explicitly selects agent-read. New benchmark measurements not launched; completed questions not rerun.
- Added detailed implementation document covering roles, pinned models/environment, IR schemas, role prefixes, pixel indexes, chunk-ID RRF, actual image transport, common budgets, CLI, no-gold input contracts, tests and AI handoff instructions. Report explicitly distinguishes question-attached images/captions (omitted) from retrieved evidence captions/images (used).
- Mem-Gallery recall diagnosis: 44/50 equal, 4 better, 2 worse; VS means both .64; no image-budget exhaustion. All 973 corpus images have provided captions.
- Validation: 42 software tests pass, including automatic delivery before forced final answer without a model read_chunk call, across all three arms and both search tools; legacy policy contrast; budget/dedup/rank integrity. No GPU model inference used for this correction.

### User authorized 100-question remeasurement after automatic transport correction
- 1000 was corrected by user to 100 per benchmark. Original M3 frozen 50+50 retained in the common 288-document corpus; Mem frozen 60 retained plus 20 VS/20 VR chosen via seeded proportional scenario/question-image sampling before new outcomes. Same prepared indexes reused, no new parsing or index builds.
- Explicit user authorization supersedes prior no-repeat rule for this policy remeasurement; still once per question per arm, no best-of. New results/three_arm_search_linked100 namespace. Qualitative cases and raw bundles will be regenerated from this policy, not reused as new evidence.
- GPU was observed idle while CPU-only selection and documentation were prepared. Related plan and implementation instructions updated.

## 2026-10-13 — search-linked100 완료 및 문서 갱신

사용자 승인된 재측정: M3DocVQA 100 × 3, Mem-Gallery 100 × 3 = 600개 성공 풀이, 문항·조건별 한 번. 모두 채점·Drive 백업 완료. 기존 인덱스 재사용. M3DocVQA EM B0/C1/G1 2/0/7%, 전달 recall 11.4/11.4/54.3%(검증 완료 70문항); Mem-Gallery EM 24/19/22%, 전달 recall 53.7/51.5/55.1%(100문항). 검색 도구 반환 이미지의 실제 전달 누락, 예산 초과, 실패, 종료 대기 모두 0. 정답률 차이의 불확실성도 보고했다.

정성 사례는 새 실행의 Aria 풍차, 고릴라 앨범, 폭스글러브 개수, 콜로세움 조명으로 교체하고 원본·전체 실행·실제 이미지·캡션·해시를 포함했다. 모두 Qwen 오답/G1 정답이지만 Mem 평균 개선을 뜻하지 않는다. Foxglove에서 Qwen 답은 3(이미지 4개), G1은 5(이미지 5개); Colosseum은 양쪽 모두 정답 이미지를 받았으며 두 후보 모두 야간이다. 캡션 없는 표지에도 픽셀 안 인쇄 문자는 있다. 기존 보고서·사례는 agent-read 역사적 기록으로 보존했다.

우리 Reader(8102, gemma-reader) 프로세스 신원을 확인하고 모든 채점·백업 완료 뒤 SIGTERM으로 중단했다. 다른 세션 GPU 프로세스는 중단하지 않았다. 최신 보고서, README, 실험 계획, 구현 명세, 현재 상태를 완료 결과에 맞게 갱신했다.

## 2026-10-14 — M3DocVQA 보고서 편집

사용자 요청으로 INTERIM_REPORT는 M3DocVQA만 다루도록 구성하고, 검색용 EmbeddingGemma 2와 답변용 Gemma 4 E4B-it을 구분했다. 텍스트 질문 벡터→텍스트/실제 이미지 인덱스 검색→RRF→실제 이미지 Reader 전달 경로, Aria의 본문 오검색과 표지 검색 차이, C1/G1 비교 의미를 명시했다. A-RAG 수정·EmbeddingGemma 2 구조 부록과 고정 설정 사본·해시를 추가했다. 고정 config sliding_window=512와 모델 카드 요약 1024의 차이를 기록했다. 정성 이미지 표는 이미지별 별도 문단으로 재구성해 임의 가로 줄바꿈을 없앴다. 문서 편집만 수행했고 추론을 재실행하지 않았다. 기존 다른 벤치마크 결과·원자료는 보존했다.

추가 사용자 요청: TrustMem(Samsung AI Center–Mountain View / Notre Dame, arXiv:2606.25161v1)을 원문 PDF와 대조해 docs/TRUSTMEM_PAPER_REVIEW.md로 정리했다. 방법·보상·TR-GRPO·학습 설정·수치·ablation·재현 한계·A-RAG 적용 제안을 구분했다. 평가 judge와 학습 Verifier의 모델명을 혼동하지 않았고, 동일 이전 상태 선호 쌍의 재현 조건과 희소 오류율 해석도 명시했다. 보고서 두 이미지 표는 1120/768px 브라우저 검증과 실제 화면 확인을 완료했다.

## 2026-10-14 — Gemini Embedding 2 논문 리뷰

사용자 지정 arXiv:2605.27295v1 원문 PDF·표를 확인하고 docs/GEMINI_EMBEDDING_2_PAPER_REVIEW.md에 구조·NCE/MRL·PFT/FT/soup·검색 결과·ablation·한계·A-RAG 적용 제안을 정리했다. 현재 측정한 EmbeddingGemma 2와 다른 모델임을 명시했다. API 계약은 2026-10-14 공식 문서로 별도 확인했다. Table 1/6 영상 기본값 차이, 이전 모델 평균 본문/표 차이, 합성 데이터 효과의 비교 기준도 기록했다. 모델 추론·API 요청·벤치마크 재실행 없이 문서만 추가했다.

## 2026-10-14 — Gemini Embedding 2 수식 설명·LaTeX

리뷰 §3–4를 보완했다. 토큰 표현·mean pooling·projection, 원문 식1의 입력/벡터 구분, cosine, NCE 분모·확률·손실, 온도, hard/in-batch negative, mask, MRL을 기호 표와 설명용 숫자 예시로 풀었다. 추가 미분·MRL 합은 유도/설명식임을 명시하고 예시 값을 실제 논문·Aria 결과와 구분했다. GitHub 지원 LaTeX inline/display 구문을 사용했다. MathJax로 display 수식16개·inline50개를 렌더링해 오류0, 숫자 계산과 1120/768px 폭 검증·스크린샷 확인 완료. 모델 추론 없음.
