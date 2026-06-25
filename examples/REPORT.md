# 예시 검증 보고서 — fresh clone에서 실제 설치·파싱·DB빌드·질의

> 독립 폴더에 GitHub `dargma/lsdrag`를 새로 clone해 **실제 API로** 전 과정을 돌린 결과.
> 목적: 의도한 flow(파싱→DB→검색)가 실제로 동작하는지, page_index 구조검색과 이미지 멀티모달 이해가
> 진짜 되는지 확인하고, 과정에서 드러난 문제를 기록한다. 실행일 2026-06-25.

## 0. 환경
- clone: `git clone https://github.com/dargma/lsdrag.git` (85 files) — 예시 PDF는 gitignore라 별도 준비.
- 전체 회귀: **9/9 통과**. `doctor --check deps --check keys`: OK (UP_TOKEN·OPENAI_API_KEY env 존재).
- 임베더: `sentence-transformers/all-MiniLM-L6-v2` **최초 실행 시 자동 다운로드(~25s)** 후 인덱싱.

## 1. 문서 준비 (분할)
ARM v8-A Reference Manual = **6,354 페이지**. 전부 빌드 아님 → 이미지+표 있는 중간 1조각만.
- figure 캡션 밀집 페이지 탐색 → p2797 부근(다이어그램 챕터) 선택.
- `split_pdf --ranges "2785-2810"` (26p) → `examples/parts/..._part1.pdf`.

## 2. 파싱 (`/rag-parse`, 실제 Upstage 호출)
```
[rag-parse] part1: 496 blocks, 7 tables, 3 figures (3 with image) → data/ir/...json
```
- **Upstage가 figure 영역을 crop해 base64로 반환** → adapter가 PNG로 저장 + IR `image_path`에 연결.
- 저장된 crop: `data/images/..._p13_e259.png`(44K), `_p17_e349.png`(96K), `_p20_e430.png`(2.1K).

## 3. DB 빌드 (인덱스 저장 형태)
IR → 임베딩 → 인덱스. 산출물(= `config.yaml`의 `paths.*`):
| 파일 | 내용 |
|------|------|
| `index/chunks.json` | `[{id, text}]` — 청크 본문 (read_chunk/keyword_search) |
| `index/sentence_index.pkl` | `{sentences, embeddings, sentence_to_chunk, chunks}` — 벡터 (semantic_search, A-RAG 호환) |
| `page_index/page_index.json` | 구조 인덱스(page·figure·table·heading + chunk_id + image_path) |
| `index/manifest.json` | doc_id → chunk_ids·image_paths (증분 add/remove용) |
| `data/images/*.png` | crop된 figure 이미지 |
- 결과: **496 chunks, 696 sentences, page_index 496 entries(figure 3)**.

## 4. 검증 — 의도한 flow가 실제로 동작하는가

### 질문 (구조 + 이미지 복합)
> “ARM exclusive memory access 상태 전이 다이어그램 — 어디에 있고 무엇을 의미하나?”

### 검색 과정 (trajectory)
1. **`page_index_search(page=17, figure_no=349)`** → hits=1, 본문 + `image_path` 동반 반환.
   (구조검색만으로 위치 특정 + 본문 확보 → read_chunk 우회)
2. **`image_read(p17_e349.png)`** → 멀티모달 Reader(GPT-4.1 mini)가 crop 이미지를 직접 해석.

### 찾은 gold + 이미지
- 위치: split-doc 기준 page 17, figure id 349 (원본 매뉴얼 ≈ p2801).
- crop 이미지: ARM **Open Access ↔ Exclusive Access 상태기계** 다이어그램.
- **VLM 이해 결과(발췌)**: “Open Access와 Exclusive Access 두 상태, LoadExcl(x,n)으로 전이,
  StoreExcl/Store/CLREX 연산, n·!n 비트 상태, Marked_address/!Marked_address …
  메모리 접근 명령을 접근 유형·비트필드 조건으로 분류하는 그림.” → **실제 다이어그램을 정확히 읽음.**
- 중복 방지: 같은 이미지 2회차 호출 시 `cached=True` (VLM 재호출 없음).

**결론: 파싱(crop 포함) → DB(인덱스+page_index+이미지) → page_index 구조검색 → 멀티모달 이미지 이해까지
의도한 flow가 실제 ARM 문서·실제 API에서 end-to-end로 동작함을 확인.**

### 4-b. agentic 루프 라이브 검증 (실제 GPT-4.1 mini가 스스로 tool 선택)
스크립트가 아닌 **진짜 LLM 주도** 루프를 라이브로 실행. 질문:
"AArch32 exclusive memory access 모델에서 Open↔Exclusive Access 전이 연산은?"
GPT-4.1 mini가 7 loop에 걸쳐 **스스로 도구를 선택**:
```
loop1 page_index_search(heading="AArch32 exclusive memory access model")
loop2 keyword_search(["Open Access","Exclusive Access", ...])
loop3 read_chunk([:239, :361])
loop4 keyword_search(["Load-Exclusive","CLREX","global/local monitor"])
loop5 read_chunk([:282, :285])
loop6 read_chunk([:361])
loop7 (tool 호출 없음 → 완료 판단, 최종 답변 생성)
```
→ Load-Exclusive로 Exclusive Access 전이, local/global monitor의 마킹 등 **근거 있는 답** 생성.
**agentic LLM 방식(자율 도구 선택 + 다단계 추론)이 실제로 동작함을 확인.**

## 5. 과정에서 드러난 문제 (개선 과제)
| # | 문제 | 영향 | 개선안 |
|---|------|------|--------|
| P1 | `split_pdf` 결과가 26p인데도 **~48MB** (pypdf가 공유 리소스를 그대로 복사) | Upstage 50MB sync 한도 근접 | **✅ 수정**: pikepdf `remove_unreferenced_resources()` → 같은 구간 **48MB→1.0MB** (라이브 확인) |
| P2 | 레지스터-인덱스 구간(5880–5930)은 **table만, figure 0** | 잘못 고르면 이미지 검증 불가 | figure는 다이어그램 챕터(예: 2785–2810). 비트필드 '레이아웃'은 table로 분류됨 |
| P3 | `page_no`가 **분할 doc 로컬 기준**; 실제 페이지는 footer "E2-2804" 형식 | 사용자가 원본 페이지로 오해 | **✅ 수정**: footer에서 `page_label` 추출. 라이브 확인 — E2-2785…E2-2792 실제 표기 획득. page_index_search가 page_label로 표시 |
| P5 | `/rag-build`가 PDF를 **재파싱**(rag-parse 캐시 IR 미사용) | Upstage 중복 호출(비용) | **✅ 수정**: `_get_ir()`가 `data/ir/*.json` 캐시를 PDF보다 최신이면 재사용 |
| P4 | figure 블록 `text`가 **raw HTML**(`<figure id=..><img alt=..>`) | 캡션 가독성↓ | (미해결) adapter에서 alt/캡션 추출해 정제 |
| P6 | ARM PDF는 **래스터 이미지 0개**(전부 벡터) | — | (긍정) Upstage가 벡터 figure도 crop·base64 제공 → image_read 정상 동작 |
| P7 | 임베더 최초 실행 시 **모델 다운로드(~25s)** | 오프라인/첫 빌드 지연 | **✅ 반영**: INSTALL에 다운로드·HF 캐시·오프라인 가이드 |
| P8 | `figure_no`가 **Upstage element id**였음 — 실제 figure 번호 아님 | 가짜 번호 시나리오 | **✅ 수정**: 캡션의 'Figure N' 추출(없으면 None). 회귀 고정 |

## 5-b. page_index 섬세 검증 (우리가 추가한 부분 — 가장 중요)

**먼저 원본 무결성 확인**: vendored A-RAG의 `agent/base.py`(loop)·`core/llm.py`·`core/context.py`·tools 3종은
upstream `a44de6b`와 **byte 단위 IDENTICAL**(diff 무차이). 우리 변경은 ① system_prompt를 파라미터로 전달
(원본 default 대체, 코드 수정 아님) ② AgentContext 서브클래스 주입뿐.

**발견(P10) — page_index가 실전에서 작동 안 했음**: Upstage가 `output_formats` 미요청 시 `content.text`/`markdown`을
빈 값으로 주고 `content.html`만 채운다. adapter가 html로 폴백 → **헤딩·본문이 raw HTML로 오염**.
그 결과 라이브 run에서 에이전트의 `page_index_search(heading="AArch32 exclusive memory access model")` → **hits=0**,
조용히 keyword_search로 폴백 → **page_index가 실제로 기여하지 못함**(침묵 실패).

**수정**: client가 `output_formats=["text","markdown","html"]` 요청 + adapter가 HTML 태그 제거(표는 markdown 구조 보존).
재파싱 결과 헤딩이 clean(`"Clear global monitor event"` 등), 표는 markdown(`| Gathering | … |`).

**재검증(라이브)**: 
- `page_index_search(heading="Store-Exclusive")` → **hits=31**, 반환 헤더 `[… page E2-2800]`(실제 페이지 라벨).
- 실제 에이전트 질의 "Store-Exclusive 동작은 몇 페이지?" → 에이전트가 **page_index_search 호출(hits=31)** →
  **1 loop**으로 *"manual page labeled E2-2800"* 정답. page_index가 read_chunk 우회하며 위치+본문 제공(설계대로).

## 6. 수정 요약 / 후속
- **✅ 수정 완료**: **P10(page_index HTML 오염 → 헤딩 매칭 복구, 가장 중요)** · P1(분할 48MB→1MB) ·
  P3(실제 page_label) · P5(IR 캐시 재사용) · P7(임베더 가이드) · P8(figure_no 라벨). P4(표 HTML)도 markdown으로 해소.
- 원본 무결성: agent loop/llm/context/tools 모두 upstream과 IDENTICAL(우리는 prompt 파라미터 + context 서브클래스만).
- 문서 현황: `rag-build --list`가 doc별 chunk·figure·table·page 표 제공.
- **남은 과제**: P2(figure 있는 구간 선택은 데이터 특성).
