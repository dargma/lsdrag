# _MASTER.md — 원본 마스터 명세 (아카이브 + 재가공 추적)

> 이 파일은 분할 전 **단일 마스터 명세**의 보관본이자, 그것을 이 repo의 11개 파일로
> 재가공하며 내린 결정의 추적 기록이다. 분할 결과물(아래 매핑)이 **실제 구현 기준**이고,
> 이 파일은 출처·의도 추적용이다.

## 분할 매핑 (마스터 §  →  파일)

| 마스터 섹션 | 배치 파일 |
|------------|-----------|
| §A | `CLAUDE.md` |
| §B | `PROGRESS.md` |
| §C0 | `docs/00_OVERVIEW.md` |
| §C1 | `docs/01_FACTS_arag.md` (불변) |
| §C2 | `docs/02_schema_IR.md` |
| §C3 | `docs/03_parser.md` |
| §C4 | `docs/04_page_index.md` |
| §C5 | `docs/05_indexing.md` |
| §C6 | `docs/06_retrieval.md` (표준 템플릿) |
| §C7 | `docs/07_agent.md` |
| §C8 | `docs/08_e2e.md` |
| §C9 | `docs/09_skill.md` |
| §D | `INSTALL.md` |
| §E | `README.md` |

## 재가공 시 마스터에서 바꾼 점 (사용자 지시 반영)

1. **`services/docling_server/` 폐기** — Upstage가 원격 API이므로 로컬 파서 서버 불필요.
   교체 경계를 `src/parser/adapter.py` **하나로 단일화**.
2. **vendor 위치 통일** — `src/agent/vendor/` (마스터의 `src/arag/`와 혼용 정리).
3. **Upstage 키 환경변수 = `UP_TOKEN`** (사용자 확정. 마스터의 `UPSTAGE_API_KEY` 대체).
4. **예시 문서 방침** — `examples/ARMv8-Reference-Manual.pdf`를 파서 소화 크기(≤100p)로 분할,
   그중 **1개 조각만** 데모로 파싱. 실제 파싱·DB 빌드 **실행은 나중에**(구조만 먼저).
5. **문서 증분 관리 추가** — 05_indexing에 `add/remove/list`(doc_id 단위), 09_skill에 사용자 명령 `docs.py`.
6. **skill 손쉬운 삭제 추가** — 09_skill에 `uninstall.py`(`--purge-data` 옵션), INSTALL에 안내.
7. **단계별 "단독 실행" 커맨드** 각 docs에 명시 — 3단계 분리를 *증명 가능*하게.
8. **개발 지시서 ↔ 배포 산출물 분리** (대원칙 5) — `CLAUDE.md`·`PROGRESS.md`·`docs/`·`_MASTER.md`·`tests/`는
   배포 skill에 포함 금지. `rag/`는 `src/`+`config.yaml`에만 의존(지시서 없이 단독 동작). config에 `engine.root` 추가.

## 사용자가 강조한 5축 (구현 내내 유지)

1. 파싱→DB→검색 **3단계 분리**(계약 2개로만 연결, 통째 교체 가능).
2. **작업 중 검증 + 사용 중 검증** 모든 모듈 필수.
3. A-RAG **불필요 요소 삭제**(런타임 import 기준 가지치기 + SOURCE.md 기록).
4. 설치·사용 **직관성**(한 명령+한 검증, `/rag` 한 줄, 사람 말 에러).
5. README = **GitHub 대표 얼굴**(직관·매력).

## 원본 마스터 전문
원본 마스터 명세 전문은 사용자 세션 프롬프트(및 상위 `/content/drive/MyDrive/CLAUDE.md` 워크플로 규칙)에
보존되어 있다. 본 repo의 11개 파일이 그 재가공·확정본이며, 충돌 시 **이 repo 파일이 우선**한다.
