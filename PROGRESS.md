# PROGRESS.md — 단일 진실의 원천

> 어떤 AI가 이어받든 이 파일만 읽으면 현황을 안다. 시작 전 읽고, 종료 시 갱신하라.

## 상태 범례
⬜ 미착수 · 🟨 진행 중 · ✅ 완료(두 스택 모두 통과) · ⛔ 막힘

## 모듈 진행표

| 모듈 | 상태 | 작업 중 검증 (Inline + 게이트) | 사용 중 검증 (런타임 가드) |
|------|------|------------------------------|---------------------------|
| `02_schema_IR`  | ✅ | 왕복 무손실 + 회귀 5건 통과 | from_dict/validate가 잘못된 입력 거부 |
| `03_parser`     | ✅ | adapter 회귀 6건(mock 응답) / non-null·figure비율 게이트 | 키없음·API오류 사람말 에러 ✓ |
| `04_page_index` | ✅ | 조회 회귀 5건 + 증분/영속화 | 빈 질의 빈결과·로드실패 안내 ✓ |
| `05_indexing`   | ✅ | 회귀 5건: 자기조회·A-RAG호환산출물·재로드·**증분add/remove·멱등** | 빈/손상 인덱스 로드실패 안내 ✓ |
| `06_retrieval`  | ✅ | G1~G4 통과(신규2종 정밀+5종 registry) | 빈결과·missing·VLM실패 폴백 ✓ |
| `07_agent`      | ✅ | FakeLLM loop완주+tool연결+이미지추적(컨텍스트 주입) | VLM 폴백(06)·LLM예외 break ✓ |
| `08_e2e`        | ✅ | 골든 3유형 전체체인 채점 ≥0.99 | 실패 시 모듈책임 분류(blame) ✓ |
| `09_skill`      | ✅ | doctor G1진단력+G2정상+G7분리 / run·docs·uninstall 껍데기 | 막힌단계서 정지+조치 ✓ |
| `10_persona`    | ✅ | 6 병렬 완주 + 종합판정(직관성 1급축) + 사망격리 | 일부 죽어도 부분결과 ✓ |
| `README.md`     | ✅ | hero 그림·실제 ARM 검색 예시·DB형태·슬래시명령 표 완비 | — |
| `eval(20케이스)` | ✅ | HW엔지니어 시나리오 20, 라이브 19/20·평균5.4loop | 단계별 trajectory 기록(`examples/EVAL_*`) |

> 현재: **문서 스캐폴딩 완료, 구현 시작 전.** 다음 작업 = `02_schema_IR`.

## 인프라 전제 (확정)

- **파서**: Upstage Document Parse API (`POST /v1/document-digitization`, `model=document-parse`).
  원격 API, 로컬 파서 서버 없음. 키 = **`UP_TOKEN`**(env). 추후 사내 파서 API로 교체(어댑터만).
- **Reader/VLM**: `reader.provider`로 선택 — **`claude_code`(기본·권장, 로컬 `claude` CLI, 키 불필요)** /
  `openai`(GPT-4.1 mini) / `anthropic`(Claude API). claude_code는 `src/agent/claude_cli.py`가 루프·VLM을 CLI로 구동.
  비교(`examples/EVAL_reader_claude_vs_gpt.md`): Claude가 다이어그램 해석 우수·환각 적음.
  멀티모달이라 VLM 겸용 → `image_read`의 별도 VLM 백엔드 불필요.
- **임베더**: A-RAG 내장 sentence-transformers. config의 `embedding.model`. 로컬 자동 로드, GPU 불필요.
- **GPU 자체 서빙 없음.** 임베더 로컬, Reader·파서는 외부 API.
- **A-RAG**: `github.com/Ayanami0730/arag` @ `a44de6b`, MIT. 필요한 모듈만 `src/agent/vendor/`로 vendoring.
- **테스트 샘플**: ARM v8-A Reference Manual에서 챕터 단위로 자른 분할 PDF 3~4개(각 50~70p).
  멀티 문서 인덱싱 + 구조 질의(레지스터 표·figure) 검증용.

## 교체 경계 (이식 시 바꿀 곳)

1. **파서 어댑터** `src/parser/adapter.py` — Upstage 응답 → IR. 사내 파서 교체 시 이것만.
2. `config.yaml`의 파서 엔드포인트·모델 (Upstage).
3. `config.yaml`의 Reader 엔드포인트·모델 (GPT-4.1 mini → 다른 멀티모달 LLM).
4. `config.yaml`의 `embedding.model` (임베더 교체).

## 개발/배포 분리 (대원칙 5)

- **개발 전용 (배포 금지)**: `CLAUDE.md`, `PROGRESS.md`, `docs/`, `_MASTER.md`, `tests/`, `examples/`, `benchmarks/`(내부 평가).
- **배포 산출물**: skill=`rag/` → `~/.claude/skills/rag/` · 엔진=`src/`+`config.yaml`+`.env` · 안내=`README.md`,`INSTALL.md`.
- `rag/` 스크립트는 `docs/`·`CLAUDE.md`를 런타임에 안 읽는다. `src/`+`config.yaml`에만 의존. (09_skill G7로 검증)

## 마스터 명세 대비 변경점 (재가공)

- `services/docling_server/` **폐기** — Upstage 원격 API라 로컬 파서 서버 불필요. 교체경계는 `adapter.py` 단일화.
- vendor 위치를 `src/agent/vendor/`로 **통일** (마스터의 `src/arag/`와 혼용 정리).
- Upstage 키 환경변수 = **`UP_TOKEN`** (사용자 확정. 마스터의 `UPSTAGE_API_KEY` 대체).
- 예시 데이터 = ARM 매뉴얼 분할 후 **이미지+테이블 있는 중간 1조각만** 파싱(전부 빌드 아님). 표지·목차 제외.

## 작업 로그 (최신이 위로)

- 2026-06-25: A-RAG vendoring 완료(`src/agent/vendor/`, 가지치기·SOURCE.md). FACTS 1~8 실코드 대조 검증(전부 일치).
- 2026-06-25: **02_schema_IR ✅** — IR 데이터클래스+무손실 왕복+런타임 가드. 회귀 `tests/test_schema.py` 5건 통과.
  (figure→image_path 하드요구는 완화: 블록단위 아닌 03 게이트의 비율 검증으로 이동.)
- 2026-06-25: **03_parser ✅** — Upstage client(키=UP_TOKEN, env에 존재 확인) + adapter(교체경계, 순수함수).
  회귀 `tests/test_parser.py` 6건(mock 응답). 런타임 가드: 키없음·API오류 사람말 에러 실증.
- 2026-06-25: **04_page_index ✅** — 구조 조회 + 증분 add/remove(멱등) + 영속화. 회귀 `tests/test_page_index.py` 5건.
- 2026-06-25: **05_indexing ✅** — IR→A-RAG호환 인덱스(chunks.json+sentence_index.pkl) + Page Index + manifest.
  임베더 주입형(실제=sentence-transformers, 테스트=가짜). `src/config.py`(config 단일로더) 추가. 회귀 5건.
  A-RAG 인덱스 형식 실코드 확인: `{sentences,embeddings,sentence_to_chunk,chunks}`, chunks.json=`[{id,text}]`.
- 2026-06-25: **06_retrieval ✅** — 신규 2종(page_index_search·image_read) BaseTool 상속, 코어 수정 0.
  vendor import shim(`src/agent/__init__.py`가 vendor를 sys.path에). 회귀 7건(G1~G4). image_read 중복방지·폴백 실증.
- 2026-06-25: **07_agent ✅** — `ImageAgentContext`(이미지추적, 서브클래스) + runner가 vendor base의 AgentContext
  참조를 monkeypatch(loop·vendor 불변). Reader=GPT-4.1mini(멀티모달, VLM겸용). 회귀 2건(컨텍스트주입+loop완주).
  ※ 테스트는 `PYTHONPATH=.` 로 실행(또는 repo 루트 cwd).
- 2026-06-25: **08_e2e ✅** — 골든셋(text/structure/image 3유형) 전체체인 채점 + 모듈책임 분류(blame).
  `tests/e2e/golden.py`(채점·분류) + `tests/test_e2e.py`(오프라인 게이트, 매칭율≥0.99). 예시=이미지+테이블 중간 1조각 방침.
- 2026-06-25: **09_skill ✅** — SKILL.md(얇은껍데기) + run/docs/doctor/uninstall + `_bootstrap`(config로 엔진 resolve).
  doctor C1~C9(이중모드 출력, 막힌단계서 정지). 회귀 4건(G1진단력·G2정상·G7패키징분리). pypdf 설치.
- 2026-06-25: **10_persona ✅** — 6 페르소나(installer/verifier/user/user2/leader/designer) 병렬 + synthesizer.
  user2=설치~사용 전반 직관성(1급 축). 회귀 3건(병렬완주·종합판정·사망격리). 종합판정: 수용, 직관성 1.00.
- 2026-06-25: **전체 9/9 모듈 회귀 통과.**
- 2026-06-25: **라이브 통합 검증** — fresh clone에서 실제 Upstage·GPT-4.1 mini로 파싱(figure crop)→DB빌드→
  page_index검색→멀티모달 image_read→**agentic 7-step 자율 루프**까지 end-to-end 동작 확인(`examples/REPORT.md`).
- 2026-06-25: **검증 중 발견 문제 수정** — P1(split 48MB→1MB, pikepdf) · P3(footer 실제 page_label, "E2-2804") ·
  P5(IR 캐시 재사용) · P7(임베더 가이드) · P8(figure_no=실제 캡션 라벨). 회귀 고정. 남은 과제 P2·P4.
- 2026-06-25: **코드 리뷰**(Google 스타일) — ruff(F,I) 우리 코드 clean, vendor 원본 보존. import 부트스트랩을
  패키지 __init__로 이동(린터 안전). 하드코딩 경로 0. 문서 현황(status) 뷰 추가. hero 그림 = gpt-image-1(최신).
- (문서 스캐폴딩 완료. 마스터 명세를 11개 파일로 분할·재가공.)
