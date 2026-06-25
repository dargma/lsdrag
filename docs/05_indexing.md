# 05 — indexing (단계 2: DB 구축 — 벡터 인덱스)

## 목적
청크 임베딩을 인덱스에 적재하고, Page Index와 함께 검색 가능한 상태로 만든다.
임베딩은 **A-RAG 내장 임베더**(sentence-transformers)를 그대로 쓴다.

## 선행 / 산출물
- 선행: 02·03·04.
- 산출물: `src/indexing/` — A-RAG 임베더 호출 + 벡터 인덱스 적재 + Page Index 영속화. 진입점 `python -m src.indexing.build`.

## 설계 결정
- **임베더는 A-RAG 내장**(사실 7). 새 임베딩 인프라 만들지 마라. config의 `embedding.model`만 지정. 별도 서빙·GPU 불필요.
- A-RAG `semantic_search`가 읽는 인덱스 형식과 **호환**되게 적재(같은 임베더여야 query/index 벡터공간 일치).
- 인덱스·Page Index는 config가 가리키는 데이터 경로에 저장(사용자 지정).
- **2↔3 계약의 나머지 절반**: 검색 단계는 이 인덱스 인터페이스만 안다.

## 증분 업데이트 (문서 추가/제거 — 필수)
사용자가 문서를 새로 넣거나 빼면 전체 재빌드 없이 해당 문서만 반영한다. **`doc_id`가 단위.**
- `add(pdf)` — 파싱→IR→임베딩→인덱스/Page Index에 그 `doc_id`만 **추가**(중복 add는 갱신=remove 후 add).
- `remove(doc_id)` — 인덱스·Page Index·이미지에서 그 `doc_id`에 속한 항목만 **제거**.
- `list()` — 현재 색인된 `doc_id`·문서명·청크 수.
- 매니페스트(`data/index/manifest.json` 등)로 doc_id↔청크/이미지 매핑을 추적해 제거 시 정확히 회수.
- 진입점: `python -m src.indexing.build --add <pdf>` / `--remove <doc_id>` / `--list`.
  (skill의 문서 관리 명령(09)이 이걸 호출한다 — 엔진 로직은 여기, 사용자 인터페이스는 skill.)

## 단독 실행 (단계 분리 증명)
```bash
python -m src.indexing.build --config config.yaml   # IR → 임베딩 → 인덱스 + Page Index 영속화
```

## 검증 — 작업 중 (Inline + 게이트)
- Inline: 소량(ARM 샘플 1조각) 적재 → 동일 텍스트로 조회 시 자기 자신이 top-1인지 확인.
- **게이트 G**: 적재 후 인덱스·Page Index가 디스크에서 재로드·조회 가능. 분할 PDF 여러 개가 `doc_id` 구분되어 누적. 소량 recall sanity 통과. → 회귀 고정.
- **게이트 G-증분**: add로 doc 추가 후 그 문서 청크가 조회됨. remove 후 그 doc_id 청크가 **인덱스·Page Index·이미지에서 모두 사라지고** 나머지는 온전. add→remove→add 멱등. → 회귀 고정.

## 검증 — 사용 중 (런타임 가드)
- 빈 인덱스/손상 → 감지 후 명확한 에러 + "build 재실행" 안내. 침묵 실패 금지.
- query 임베더 ≠ index 임베더(모델 불일치) 감지 시 경고.
