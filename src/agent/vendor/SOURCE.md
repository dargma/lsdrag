# vendor/SOURCE.md — A-RAG 출처·가지치기 기록

## 출처
- 레포: `github.com/Ayanami0730/arag`
- 커밋: `a44de6b`
- 라이선스: **MIT** (vendoring 자유)
- 가져온 날: 2026-06-25
- FACTS 대조: `docs/01_FACTS_arag.md`의 사실 1~8을 실코드와 대조 검증 완료(전부 일치).

## 가져온 파일 (런타임 경로가 import하는 것만)
```
arag/__init__.py
arag/core/{__init__,context,llm,config}.py
arag/tools/{__init__,base,registry,semantic_search,keyword_search,read_chunk}.py
arag/agent/{__init__,base}.py
```

## 삭제(미포함) 파일 + 이유 (대원칙 3 — 불필요 요소 삭제)
| 미포함 | 이유 |
|--------|------|
| `scripts/eval.py` | 평가 전용. 런타임 경로 아님. |
| `scripts/batch_runner.py` | 배치 평가 전용. 런타임 경로 아님. |
| `scripts/build_index.py` | 우리 `src/indexing/`가 대체. 인덱스 빌드는 우리 모듈 책임. |
| `tests/test_import.py`, `tests/__init__.py` | 상류 테스트. 우리 `tests/`가 대체. |

판단 기준: **"우리 파이프라인(파싱→DB→검색)의 런타임 경로가 import하는가?"** 아니면 제외.

## 수정 정책
- vendor 직접 수정 **금지**. 필요 시 바깥에서 상속/확장(`src/agent/`).
- 유일한 코어 확장: `AgentContext`에 이미지 추적 추가 → **서브클래스로**(`src/agent/`), vendor 파일은 원본 유지.
- 부득이 vendor를 고치면 이 파일에 변경점·이유·줄을 기록.
