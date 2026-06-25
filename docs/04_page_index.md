# 04 — page_index (단계 2: DB 구축 — 구조 인덱스)

## 목적
구조 메타데이터(제목·페이지·Figure·날짜·이미지명)를 **질의 가능한 Page Index**로 구축. 구조적 질의를 커버.
그래프 없이 GraphRAG 효과를 내는 핵심 축.

## 선행 / 산출물
- 선행: 02(IR), 03(파서 산출 IR).
- 산출물: `src/page_index/` — IR에서 구조 인덱스 생성 + 조회 인터페이스 + 디스크 영속화.

## 설계 결정
- 그래프 노드가 아니라 **질의 가능한 메타데이터 인덱스**(그래프 불필요가 이 프로젝트의 전제).
- 조회 키: page / figure_no / heading / doc_id (부분 지정 허용).
  `page`는 split-local 번호 **또는 인쇄된 페이지 라벨**(예 `E2-2804`)로 조회 가능 — 에이전트가 결과에서 본 라벨로 다시 조회할 수 있게.
- 각 엔트리는 **청크 ID·image_path·page_label**을 보유 → `page_index_search`가 본문+이미지명+실제 페이지 표기를 한 번에 반환.
- **2↔3 계약의 절반**: 검색 단계는 이 조회 인터페이스만 알면 된다(내부 저장 형식 비공개).

## 단독 실행 (단계 분리 증명)
```bash
python -m src.page_index.build --ir data/ir/ --out data/page_index/
python -m src.page_index.query --page 3 --figure 2     # 기대 청크 ID 출력
```

## 검증 — 작업 중 (Inline + 게이트)
- Inline: IR 1건 → 인덱스 구축 → "page=3, figure=2" 조회 시 기대 청크 ID assert.
- **게이트 G**: 라벨된 구조 질의 set에서 (page)/(page+figure)/(제목) 조회 정확도 임계 이상. 존재하지 않는 위치 → **빈 결과(예외 아님)**. → 회귀 고정.

## 검증 — 사용 중 (런타임 가드)
- 인덱스 미빌드/로드 실패 → 명확한 에러 + "build 먼저 실행" 안내.
- 없는 page/figure 조회 → 빈 결과 + 친절한 메시지. 크래시 금지.
