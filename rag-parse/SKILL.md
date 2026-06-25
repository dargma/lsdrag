---
name: rag-parse
description: >-
  문서 파싱 점검. PDF 1개를 파서(Upstage)로 IR(중간표현)로 변환해 결과를 확인한다.
  "이 PDF 파싱해줘", "파싱 결과 보여줘", "문서 구조(표/그림) 추출 확인" 같은
  빌드 전 점검 요청에서 발동. DB 빌드는 /rag-build, 질의응답은 /rag.
---

# rag-parse — 파싱만 (문서 → IR 점검)

얇은 껍데기. 엔진(`src.parser`)을 호출해 1개 PDF를 IR로 변환·저장한다.

```
python ${CLAUDE_SKILL_DIR}/scripts/run.py <pdf> [--out data/ir]
```

출력: 블록 수 / figure 수 / image_path 채워진 비율, IR JSON 경로.
표·그림이 든 본문 조각인지 빌드 전에 눈으로 확인하는 용도. 키는 env(`UP_TOKEN`).
