# LSD-RAG 검증 리포트 — 무엇을 풀고, 어떤 단계로, 왜 효과적인가

> 대상 독자: 이 프로젝트를 처음 보는 엔지니어·리뷰어.
> 이 문서 하나로 "어떤 문제를 푸는가 → 어떻게 푸는가 → 정말 되는가(라이브 증거) → 왜 효과적인가"를 파악할 수 있다.
> 원시 로그는 [`REPORT.md`](REPORT.md)(설치·빌드 검증)와 [`EVAL_pageindex_agent.md`](EVAL_pageindex_agent.md)(20 케이스 trajectory)에 있다.

---

## 0. TL;DR

- **푸는 문제**: 수천 페이지 기술 매뉴얼에서 *"답이 몇 페이지·어느 그림·어느 표에 있는지"*까지 찾고, 다이어그램까지 읽어 답한다.
- **방법**: 파싱(그림 crop 포함) → DB(벡터 + Page Index) → **에이전트가 도구를 스스로 골라** 위치를 짚고 그림을 읽는다.
- **증거**: fresh clone에서 **실제 Upstage·GPT-4.1 mini로 end-to-end 동작 확인**. HW 엔지니어 시나리오 20개 중 **19개** 의도대로, 평균 5.4 도구호출.
- **엄밀성**: 검증 중 *실제로 동작하지 않던* 결함(특히 page_index가 침묵 실패)을 trajectory 기록으로 잡아 고치고 재검증했다.

---

## 1. 어떤 문제를 푸는가

| 현장의 문제 | 일반 RAG의 한계 | LSD-RAG의 해결 |
|------------|----------------|----------------|
| "이 레지스터 비트필드가 **몇 페이지 어느 그림**에 있지?" | 비슷한 문장 조각만 반환, 위치 모름 | **Page Index**로 페이지·figure·표를 짚어 인용 |
| 비트필드/상태 다이어그램이 **그림**이라 텍스트로 안 잡힘 | 이미지 무시 | **멀티모달 image_read**가 그림을 읽어 설명 |
| 표가 **구조**(행·열)인데 평문으로 뭉개짐 | 표 구조 손실 | 표를 **markdown 구조**로 보존 |
| 한 질문에 **그림 + 표**가 같이 필요 | 한 모달리티만 | 한 질의에서 **둘 다 불러와 통합** |
| 사내 파서/모델로 **갈아끼우기** 어려움 | 모놀리식 | **파싱→DB→검색 3단계 분리**, 부품 교체 |

---

## 2. 어떻게 푸는가 (단계)

```
[1 파싱]  문서 → Upstage → IR        (그림은 crop해 이미지 저장, footer에서 실제 페이지 라벨 추출)
[2 DB]    IR → 임베딩 + Page Index   (page/figure/table 메타 + 이미지 + 실제 페이지 라벨)
[3 검색]  질문 → 에이전트가 도구 선택 → 위치 짚고 그림 읽어 답
```

검색 단계의 핵심은 **에이전트가 단계별로 도구를 고르는 것**이다(도구 5종):

| 도구 | 역할 |
|------|------|
| `semantic_search` / `keyword_search` | 토픽으로 후보 찾기 |
| `read_chunk` | 청크 전체 본문 |
| `page_index_search` | **구조 질의** — 위치 특정 + 본문 + 이미지명 + 실제 페이지 라벨(우리가 추가) |
| `image_read` | **그림을 멀티모달로 읽기**(우리가 추가) |

> 도식: 저장소 최상단 `docs/hero.png` — 사용자 시작점 → 에이전트 다단계 도구 호출 → 근거 답.

---

## 3. 정말 되는가 — 라이브 증거

### 3.1 환경
- GitHub `dargma/lsdrag`를 **새로 clone**(예시 PDF는 별도 준비). 회귀 9/9, `doctor` 통과.
- 데이터: ARM v8-A Reference Manual(**6,354p**)에서 그림+표가 있는 중간 1조각(E2-2785~E2-2810, 26p)만 사용.
- 실제 API: 파서 Upstage, Reader/VLM GPT-4.1 mini, 임베더 sentence-transformers(로컬).

### 3.2 파이프라인 한 줄 흐름
- 파싱: **494 blocks, 7 tables, 3 figures** — 그림은 base64로 crop돼 PNG로 저장, IR에 image_path 연결.
- 빌드: 벡터 인덱스 + Page Index + manifest. 같은 임베더라 검색공간 일치.

### 3.3 예시 ① — 구조 질의(Page Index가 read_chunk를 우회)
> 질문: "Store-Exclusive와 글로벌 모니터 동작은 **몇 페이지**에 있나?"

```
loop1: page_index_search(heading="Store-Exclusive instruction")  → hits=31
       → [doc page E2-2800] 본문 동반 반환
loop2: (도구 호출 없음) 최종 답변
답변: "described on manual page labeled E2-2800."
```
**1 loop**으로 위치+본문을 얻어 답한다(page_index가 본문을 함께 주므로 read_chunk 생략). 인용은 **문서에 인쇄된 실제 페이지 라벨**.

### 3.4 예시 ② — 멀티모달(그림 + 표 동시)
- 직접: crop한 figure 이미지 + 표 markdown을 한 번의 멀티모달 호출로 주면, GPT-4.1 mini가
  이미지의 두 상태("Open/Exclusive Access")와 표 첫 행("Gathering")을 **둘 다 읽고 관계까지** 설명.
- 에이전트 경로: `page_index_search(page='E2-2801')` → 같은 페이지 figure 이미지 노출 →
  **`image_read`로 다이어그램 읽기** + 표 검색 → 그림과 표를 통합한 답.

### 3.5 평가 — HW 엔지니어 시나리오 20개
스핀락 설계·모니터 클리어·메모리 컨트롤러(Gathering)·저전력(WFE/SEV)·검증용 상태기계 등 **실무 질문 20개**를
실제로 돌려 단계별 trajectory와 "의도대로 도구를 골랐는가"를 기록.
- **결과: 19/20 의도 충족, 평균 5.4 도구호출.** (케이스별 단계 기록은 `EVAL_pageindex_agent.md`)

---

## 4. 왜 효과적인가 (설계 근거)

1. **Page Index가 위치+본문+이미지를 한 번에** → 구조 질의가 1~2 loop으로 끝나 토큰·지연 절감.
2. **실제 인쇄 페이지 라벨로 인용**(E2-2800 등) → 사람이 바로 문서에서 찾아 검증 가능(신뢰).
3. **그림을 못 찾고 끝나지 않게** → 같은 페이지 figure 자동 노출 + 무번호 그림도 토픽 검색에 잡히는 합성 캡션 → `image_read` 트리거.
4. **토큰 효율** → page_index_search가 엔트리 ≤6·본문 ≤500자·이미지 ≤4로 캡(페이지 전체 덤프 방지), 프롬프트가 "최소 호출".
5. **원본 엔진 무손상** → A-RAG의 agent loop/LLM/context/도구는 upstream과 **byte-identical**, 우리는 system_prompt(파라미터) + context 서브클래스만 추가 → 검증된 루프 위에 얹음.

---

## 5. 엄밀성 — "발견하고 고친" 실제 문제들

이 프로젝트의 신뢰성은 *깨끗한 데모*가 아니라 **결함을 trajectory로 잡아 고친 기록**에 있다.

| # | 발견(어떻게 알았나) | 영향 | 수정 → 재검증 |
|---|---------------------|------|---------------|
| **P10** | page_index_search(heading) hits=0, 에이전트가 조용히 keyword로 폴백 | **page_index가 실전에서 미작동**(침묵 실패) | Upstage가 html만 줘서 헤딩 오염 → `output_formats` 요청 + HTML 제거. 재검증 hits=31 |
| **P8** | figure_no가 Upstage element id(349)였음 | "page X, figure Y"가 가짜 번호 | 캡션의 실제 'Figure N' 추출(없으면 None) |
| **P3** | page_no가 split-local | 원본 페이지 오인 | footer에서 실제 라벨(E2-2800) 추출, 검색·인용에 사용 |
| **P1** | 26p split이 48MB | Upstage 50MB 한도 근접 | pikepdf로 미참조 리소스 제거 → **48MB→1MB** |
| **P5** | 빌드가 PDF 재파싱 | Upstage 중복 호출(비용) | IR 캐시 재사용 |
| **figure 발견성** | 20케이스에서 image_read 미트리거 | 그림을 안 읽고 텍스트로만 답 | 합성 캡션 + 같은 페이지 figure 자동 노출 → 재평가 16→19/20 |
| **인용 정확성** | 에이전트가 chunk id를 페이지로 오인 | 가짜 인용 | 프롬프트: 인쇄 라벨만 인용 |

---

## 6. 한계 / 정직한 경계

- **무번호 figure**: 문서가 그림에 번호를 안 매기면 번호로 못 찾는다(페이지·토픽으로 우회). 
- **비결정성**: 텍스트로 답이 충분하면 에이전트가 `image_read`를 생략하기도 한다(답은 정확). 항상 강제하면 토큰 효율과 충돌하므로 균형 유지.
- **데이터 의존**: 적절한 figure+표가 있는 구간을 골라야 멀티모달이 의미 있다.
- 평가는 단일 문서(ARM 1조각) 기준 — 멀티문서·타 매뉴얼 확장은 일반화 설계(다중 페이지 포맷 등)로 열어둠.

---

## 7. 재현 방법

```bash
git clone https://github.com/dargma/lsdrag.git && cd lsdrag
export UP_TOKEN=...  OPENAI_API_KEY=...
python scripts/split_pdf.py --input <manual.pdf> --ranges "<figure+table 구간>" --out examples/parts/
cp examples/parts/<part>.pdf data/docs_in/
python -m src.indexing.build --config config.yaml          # 파싱→DB
python benchmarks/eval_pageindex_agent.py                       # 20 케이스 라이브 평가 → EVAL_*.md
```

---
*상세 단계 로그: [`REPORT.md`](REPORT.md) · [`EVAL_pageindex_agent.md`](EVAL_pageindex_agent.md)*
