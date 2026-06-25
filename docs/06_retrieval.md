# 06 — retrieval (단계 3: 검색 — tool 5종) · 모듈 명세 표준 템플릿

> **이 문서가 모듈 명세의 표준 형식이다.** 다른 모듈도 이 구성을 따른다:
> 목적 / 선행·산출물 / 설계 결정 / 단독 실행 / **검증 — 작업 중(Inline+게이트)** / **검증 — 사용 중(런타임 가드)**.

## 목적
Agent가 호출할 검색 tool 5종 제공. A-RAG 기존 3종을 vendor로 쓰고, 신규 2종을 추가한다.

| Tool | 출처 | 역할 |
|------|------|------|
| `keyword_search`    | A-RAG(vendor) | 어휘 매칭 → 스니펫 |
| `semantic_search`   | A-RAG(vendor) | 임베딩 유사도 → 스니펫 |
| `read_chunk`        | A-RAG(vendor) | 청크 전체 본문 |
| `page_index_search` | **신규** | 구조 질의 → 위치 특정 + 본문 + 이미지명 |
| `image_read`        | **신규** | 이미지명 → VLM 해석 → 텍스트 설명(조건부) |

## 선행 / 산출물
- 선행: 02(IR), 05(인덱스+Page Index), 01_FACTS(tool 계약).
- 산출물: `ToolRegistry`에 5종 등록. 각 tool은 BaseTool 계약 충족.

## 설계 결정
1. 신규 tool도 `BaseTool` 상속, `registry.register()`로 추가(사실 1·2). **코어 수정 금지.**
2. `page_index_search`는 예외적으로 **위치 특정 + 본문 + 이미지명 + 페이지 라벨**을 함께 반환(사실 3) → 구조 질의에서 read_chunk 우회.
   - **토큰 효율**: 한 호출 반환 = 엔트리 ≤6(`top_k`)·본문 ≤500자 truncate·이미지 ≤4. 페이지 전체 덤프 방지.
   - **page 라벨 조회**: `page`는 로컬 번호 또는 인쇄 라벨(`E2-2804`) 모두 매칭.
   - **figure 자동 노출**: 텍스트 hit와 같은 페이지의 figure 이미지를 surface → 무번호 그림도 image_read로 읽게.
   - **빈 결과**: 막다른 메시지 대신 다음 한 걸음(토픽 검색/짧은 헤딩) 안내.
3. `image_read`는 **조건부**(텍스트로 불충분할 때만, 스키마 설명에 명시) + **중복 방지**(사실 4 패턴 복제).
4. **VLM 별도 백엔드 없음** — Reader(GPT-4.1 mini)가 멀티모달이라 이미지를 직접 처리(사실 8).
   구현 택1: (a) 이미지를 Reader에 직접 전달(멀티모달 메시지), (b) Reader로 이미지를 텍스트 설명화해 기존 `(str,dict)` 계약 유지.
   **loop 변경 최소화를 원하면 (b).**

## 단독 실행 (단계 분리 증명)
```bash
python -m src.retrieval --selftest   # mock context로 5종 execute가 (str,dict) 반환하는지
```

## 검증 — 작업 중 (Inline + 게이트)
- Inline: 각 신규 tool 작성 즉시 샘플 1건으로 `execute`가 `(str,dict)` 반환 assert. `get_all_schemas()`에 5개 잡히는지. `image_read` 같은 이미지 2회 호출 시 2번째 VLM 재호출 없음(mock).
- **G1 계약(유닛)**: 5종 모두 BaseTool, `get_all_schemas()` 유효 스키마 5개, `execute` 반환 `(str,dict)`.
- **G2 page_index_search(유닛)**: (page)/(page+figure)/(없는 위치→빈 결과). 본문+이미지명 동반.
- **G3 image_read(유닛)**: 중복 시 VLM 재호출 없음(mock 호출 횟수 검증) = 경계 분리 증거.
- **G4 통합**: 5종 등록 registry로 mock LLM이 각 1회 호출 시나리오 완주, retrieval_log에 기록.

## 검증 — 사용 중 (런타임 가드)
- 인덱스/Page Index 비었거나 로드 실패 → 명확한 에러 + 조치 안내.
- tool `execute`가 빈 결과/예외 → agent에 의미 있는 메시지 반환(크래시 금지).
- `image_read`: image_path 없음/깨진 파일 → graceful(텍스트만으로 진행).
- VLM(Reader) 호출 실패 → 폴백(텍스트 컨텍스트만으로 답변 시도) + 로그.

## 다음
G1~G4 + 작업 중·사용 중 검증 통과 후 07_agent로.
