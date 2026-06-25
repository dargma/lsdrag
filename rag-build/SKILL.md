---
name: rag-build
description: >-
  RAG 인덱스(DB) 빌드·갱신. 사내/기술 문서 PDF를 파싱해 검색 가능한 인덱스로 만든다.
  "문서 인덱싱/DB 빌드/색인 갱신", "이 PDF 추가해줘", "X 문서 빼줘", "색인된 문서 목록"
  같은 요청에서 발동. 질의응답은 /rag, 파싱 점검은 /rag-parse.
---

# rag-build — DB 빌드 / 문서 관리

얇은 껍데기. 엔진(`src.indexing.build`)을 호출한다.

- 전체 빌드: `python ${CLAUDE_SKILL_DIR}/scripts/run.py`
- 문서 추가: `python ${CLAUDE_SKILL_DIR}/scripts/run.py --add <pdf>`
- 문서 제거: `python ${CLAUDE_SKILL_DIR}/scripts/run.py --remove <doc_id>`
- 목록:     `python ${CLAUDE_SKILL_DIR}/scripts/run.py --list`

빌드는 내부에서 **파싱(Upstage)→IR→임베딩+Page Index→인덱스 저장**을 수행한다.
경로·모델은 `config.yaml`, 키는 env(`UP_TOKEN`,`OPENAI_API_KEY`). 실패 시 doctor 조치를 사람 말로 전달.
