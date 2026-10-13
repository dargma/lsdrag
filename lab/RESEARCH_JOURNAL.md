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
