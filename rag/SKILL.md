---
name: rag
description: >-
  사내/기술 문서 질의응답 엔진. PDF 매뉴얼·스펙·레퍼런스에서 답을 찾고,
  "몇 페이지/어느 figure/어느 표에 있나" 같은 구조 질의와 비트필드 다이어그램 등
  이미지 해석까지 처리한다. 문서 검색·기술 매뉴얼 질문·레지스터/명령어/표/그림을
  묻는 요청, "이 PDF 추가/제거" 같은 문서 관리 요청에서 발동.
---

# rag — 그래프 없는 agentic RAG

이 skill은 **얇은 껍데기**다. 모든 로직은 엔진(`src/`)에 있고, 여기서는 스크립트만 호출한다.

## 질의
사용자가 사내/기술 문서에 대해 물으면:
```
python ${CLAUDE_SKILL_DIR}/scripts/run.py "<질문>"
```
출력(답변 + 출처 페이지)을 사용자에게 전달한다.

## 문서 관리
- 추가: `python ${CLAUDE_SKILL_DIR}/scripts/docs.py add <pdf>`
- 제거: `python ${CLAUDE_SKILL_DIR}/scripts/docs.py remove <doc_id>`
- 목록: `python ${CLAUDE_SKILL_DIR}/scripts/docs.py list`

## 설치 점검 / 삭제
- 점검: `python ${CLAUDE_SKILL_DIR}/scripts/doctor.py --json`  (C1~C9, 막힌 단계+조치 출력)
- 삭제: `python ${CLAUDE_SKILL_DIR}/scripts/uninstall.py`  (데이터까지: `--purge-data`)

## 규칙
- 경로·모델은 `config.yaml`, 키는 env(`UP_TOKEN`, `OPENAI_API_KEY`). 평문 노출 금지.
- 실패 시 스택트레이스만 던지지 말고 doctor가 낸 "조치"를 사용자에게 사람 말로 전달.
