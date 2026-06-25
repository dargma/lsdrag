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
| P1 | `split_pdf` 결과가 26p인데도 **~48MB** (pypdf가 공유 리소스를 그대로 복사) | Upstage 50MB sync 한도에 근접, 100p 분할 시 초과 위험 | 분할 시 content stream 압축 / `pikepdf` 리소스 subset / 페이지수 더 작게 |
| P2 | 레지스터-인덱스 구간(5880–5930)은 **table만, figure 0** | 잘못 고르면 이미지 검증 불가 | figure는 다이어그램 챕터(예: 2785–2810). 비트필드 '레이아웃'은 table로 분류됨 |
| P3 | `page_no`가 **분할 doc 로컬 기준**(p17 = 원본 ≈2801); 실제 페이지는 footer "E2-2804" 형식 | 사용자가 원본 페이지로 오해 | footer 패턴을 page_label로 추출해 표시(후속) |
| P8 | `figure_no`가 **Upstage element id**(예 349)였음 — 문서의 실제 figure 번호 아님 | "page X, figure Y" 시나리오가 가짜 번호 | **수정 완료**: 캡션의 실제 'Figure N' 라벨 추출(없으면 None). 이 ARM 상태도는 번호 없는 inline 그림 → page로만 조회. 번호 시나리오는 캡션 있는 figure로 시연 |

> 후속 반영: P8(figure_no 실제 라벨화)은 `src/parser/adapter.py`에서 완료, `tests/test_parser.py`에 회귀 고정.
| P4 | figure 블록 `text`가 **raw HTML**(`<figure id=..><img alt=..>`) | 캡션 가독성↓ | adapter에서 alt/캡션 추출해 정제 |
| P5 | `/rag-build`(cmd_build)가 **PDF를 재파싱** (rag-parse 캐시 IR 미사용) | Upstage 중복 호출(비용) | 빌드가 `data/ir/*.json` 캐시를 우선 사용하도록 |
| P6 | ARM PDF는 **래스터 이미지 0개**(전부 벡터) | — | (긍정) Upstage가 벡터 figure도 crop·base64 제공 → image_read 정상 동작 |
| P7 | 임베더 최초 실행 시 **모델 다운로드(~25s)** | 오프라인/첫 빌드 지연 | INSTALL에 다운로드·캐시 가이드 추가(아래) |

## 6. 권장 후속
- P1·P5는 빌드 효율/안정성 직결 → 우선 개선.
- 문서 add/remove 현황은 `rag-build --list`를 **풍부한 status**(doc별 chunk·figure·table·페이지·갱신시각)로 확장.
