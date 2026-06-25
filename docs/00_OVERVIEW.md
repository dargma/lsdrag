# 00 — OVERVIEW

## 한눈에 보는 구조 (3단계 엄격 분리)

각 단계는 독립 교체 가능(다른 파서/임베더/검색기로 통째 바꿔치기). 단계 간 연결은 **명시적 계약**만.

```
┌─ 1. 파싱 ────────┐   ┌─ 2. DB 구축 ──────────┐   ┌─ 3. 검색 ──────────────────┐
│ 문서 → Upstage   │ → │ IR → 임베딩 → 인덱스    │ → │ 쿼리 → A-RAG agent          │
│      → IR        │IR │     → Page Index       │   │ → tool 5종 → [VLM] →        │
│                  │   │                        │   │ Reader(GPT-4.1 mini) → 답   │
└──────────────────┘   └────────────────────────┘   └────────────────────────────┘
   교체: 파서             교체: 임베더/저장소           교체: 검색전략/Reader
   계약: IR (02)          계약: 인덱스/PageIndex 인터페이스
```

각 단계는 **단독 실행·테스트 가능**:
- 1만: `python -m src.parser ...` → IR(JSON) 산출물 눈으로 확인.
- 2만: IR 입력 → 인덱스/PageIndex 디스크 산출 → 재로드 조회.
- 3만: 인덱스 입력 → 쿼리 → 답변.

## 데이터 흐름의 핵심

- **IR(중간표현) = 1↔2 계약**, **인덱스/Page Index 인터페이스 = 2↔3 계약**. 단계는 서로의 내부를 모른다.
- **그래프 없이** vector + 구조 메타데이터(Page Index) + 조건부 VLM으로 GraphRAG 효과를 낸다.
- 확장은 **tool 추가**로. A-RAG의 `BaseTool` 계약에 `page_index_search`, `image_read`를 더한다.

## 폴더 구조

```
lsdrag/
├── CLAUDE.md · PROGRESS.md · INSTALL.md · README.md
├── config.yaml                # 경로·모델·엔드포인트 (키는 env, 평문 금지)
├── .env.example               # UP_TOKEN, OPENAI_API_KEY
├── docs/                      # 명세 (이 문서들) + _MASTER.md
├── src/
│   ├── schema/                # IR (02)
│   ├── parser/                # Upstage 클라이언트 + IR 어댑터 (교체 경계) (03)
│   ├── page_index/            # 구조 메타 인덱스 (04)
│   ├── indexing/              # A-RAG 임베더 호출 + 인덱스 적재 (05)
│   ├── retrieval/             # tool 5종 (06)
│   └── agent/vendor/          # A-RAG 가져온 부분 (가지치기됨, 07)
├── rag/                       # skill (SKILL.md, scripts/, doctor.py) (09)
├── scripts/split_pdf.py       # ARM 매뉴얼 분할
└── tests/                     # Inline 회귀 + E2E (08)
```

## 두 개의 교체 경계 (이식 포인트)

1. **파서** — `src/parser/adapter.py`(Upstage 응답→IR). 사내 파서 교체 시 어댑터만. (원격 API라 로컬 서버 없음.)
2. **Reader/VLM** — config의 reader 블록. GPT-4.1 mini가 멀티모달이라 VLM 겸용(별도 백엔드 없음). 다른 멀티모달 LLM 교체 시 config만.

## 두 스택 검증 (모든 모듈)

- **작업 중**: 인라인 assert → `tests/` 회귀 + Post-hoc 게이트(완료 정의).
- **사용 중**: 런타임 가드, 침묵 실패 금지. `doctor.py`가 설치판.
