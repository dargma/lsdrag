# 03 — parser (단계 1: 파싱)

## 목적
문서를 IR로 변환. **Upstage Document Parse API**를 호출해 구조화 elements를 받아 IR로 매핑한다.
Upstage가 이미 원격 API이므로 **로컬 파서 서버를 만들지 않는다**(마스터의 docling_server 폐기).

## 선행 / 산출물
- 선행: 02_schema_IR.
- 산출물: `src/parser/client.py`(Upstage 호출), `src/parser/adapter.py`(Upstage 응답→IR). `scripts/split_pdf.py`.

## 설계 결정
- **API 경계 격리(교체 경계)**: 클라이언트는 Upstage 응답 JSON 계약에만 의존. 사내 파서 교체 시 **`adapter.py`만** 바뀐다.
- 엔드포인트·모델·키는 config/env에서 읽는다. **키 평문 금지** — `UP_TOKEN`(env).
- Upstage→IR 매핑(사실 8): `heading1`→heading, `figure`/`chart`→block_type=figure, `table`→table,
  `caption`→caption, 나머지→text. `page`→page_no, `coordinates`→bbox.
- **이미지 crop은 figure/chart만**: `base64_encoding=['figure','chart']`로 받아 파일 저장 + IR `image_path`.
  표는 이미지로 안 남긴다(HTML/markdown 구조로 chunk에 보존 — image_read 불필요).
- **clean text 추출**: Upstage가 `content.html`만 채우는 경우가 있어 `output_formats=['text','markdown','html']`를
  요청하고, adapter가 `text>markdown>HTML태그제거` 순으로 **깨끗한 텍스트**를 뽑는다(표는 markdown 구조 보존).
  HTML 오염은 page_index 헤딩 매칭을 망가뜨리므로 필수.
- **page_label(P3)**: footer/header에서 문서에 인쇄된 페이지 표기를 추출해 `ParsedBlock.page_label`에 채운다.
  다중 포맷 일반화(예 `E2-2804`/`D1-1234`/`12-3`/`Page 45`/순수숫자). split-local `page_no`와 별개.
- **figure_no = 실제 캡션 라벨**: Upstage element id가 아니라 캡션의 'Figure N'을 쓴다(없으면 None).
- **figure 검색성**: 무번호 figure는 같은 페이지 헤딩·본문 스니펫 + 페이지 라벨로 **합성 캡션**을 넣어
  semantic/keyword 검색에 노출(→ 에이전트가 image_read 트리거).
- 분할 PDF 다문서: 각 PDF가 별도 `doc_id`. sync 100p 한도 안(50~70p)이라 async 불필요.

## 예시 문서 처리 방침 (ARM 매뉴얼)
- 예시 PDF = `examples/ARMv8-Reference-Manual.pdf` (~5000p+, 39MB). **통째로 파싱하지 않는다.**
- **분할 우선**: `scripts/split_pdf.py`로 **파서가 소화 가능한 크기**(Upstage sync ≤100p, 권장 50~70p)로 자른다.
  "4등분" 개념이되 한 조각이 100p를 넘으면 더 잘게(예: 챕터/페이지 범위 단위) 나눈다. 각 조각 = 별도 `doc_id`.
- **예시는 1개 조각만**: 분할 후 데모용으로 **딱 1개 조각만** 파싱한다(전권 아님, 전부 빌드 아님).
- **그 1개는 반드시 이미지(figure)와 테이블이 들어있는 중간 본문 조각**을 고른다.
  앞쪽 표지/목차/서문(TOC·preface)은 표·그림이 없어 구조/이미지 질의 검증이 안 된다.
  ARM 매뉴얼이면 시스템 레지스터 표·비트필드 figure가 나오는 **중간 챕터 범위**를 선택.
- 선택 방법: 분할 후 각 조각을 빠르게 훑어(또는 1조각 파싱해 IR의 block_type 분포 확인)
  `table`·`figure` 블록이 함께 있는 조각 1개를 데모로 확정. 그것만 `data/docs_in/`에 넣는다.
- 멀티문서 확장은 나중에. **지금은 구조(코드·스크립트·계약)만**, 실제 파싱·빌드 실행은 키 준비 후.

## 단독 실행 (단계 분리 증명)
```bash
# 1) 파서가 소화 가능한 크기로 분할 (한 조각 ≤100p)
python scripts/split_pdf.py --input examples/ARMv8-Reference-Manual.pdf \
    --ranges "1-60,61-120,121-180,181-240" --out examples/parts/
# 2) 그중 1개만 파싱 → IR(JSON) 산출, 눈으로 확인 (실제 실행은 나중에)
python -m src.parser --pdf examples/parts/part1.pdf --out data/ir/
```

## 검증 — 작업 중 (Inline + 게이트)
- Inline: ARM 분할 PDF 1개 → Upstage 호출 → elements 수신 → IR 변환 즉시 출력. figure 페이지에서 `image_path` 채워지는지 확인. 키 없을 때(env 미설정) 명확한 에러 확인.
- **게이트 G**: ARM 분할 PDF N개에서 `page_no`·본문 non-null 비율 임계 이상. figure 페이지 `image_path` 채워짐. `doc_id` 문서별 구분. → 회귀 고정(실제 API는 **mock 응답으로도 한 벌** 고정).

## 검증 — 사용 중 (런타임 가드)
- `UP_TOKEN` 없음/빈 값 → "키를 .env에 설정하세요" 명확 에러(스택트레이스 금지).
- Upstage 4xx/5xx·타임아웃 → 어떤 파일에서·왜 실패했는지 메시지 + 재시도/건너뛰기 안내.
- 파서 응답이 IR 계약을 못 채우면(빈 elements 등) 침묵 통과 금지, 경고 + 해당 doc 표시.
