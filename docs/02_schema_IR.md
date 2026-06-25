# 02 — schema_IR (단계 1↔2 계약)

## 목적
파서 종류와 무관한 **공통 중간표현(IR)**. 모든 다운스트림의 단일 계약. 파서가 무엇이든 출력은 IR로 흡수된다.

## 선행 / 산출물
- 선행: 없음(최초 모듈).
- 산출물: `src/schema/`의 IR 데이터클래스 + 직렬화(JSON 무손실 왕복).

## 설계 결정
- 최소 필드:
  - `ParsedBlock{ text, page_no, heading, block_type, figure_no, image_path, bbox? }`
  - `ParsedDoc{ doc_id, title, date, blocks[] }`
- `block_type`은 최소 집합 `"text" | "table" | "figure" | "caption"`으로 시작, 확장 가능하게.
- 이미지·구조 필드(`figure_no`, `image_path`, `page_no`)는 **필수 후보** — Page Index(04)가 의존.
- JSON 직렬화/역직렬화 **무손실**(파서 API 응답이 이 형식으로 떨어진다).

## 단독 실행 (단계 분리 증명)
```bash
python -m src.schema --selftest   # 더미 ParsedDoc 왕복 후 동일성 출력
```

## 검증 — 작업 중 (Inline + 게이트)
- Inline: 더미 ParsedDoc 1건 → `to_json → from_json` 동일성 assert. 필수 필드 누락 시 명확한 에러 즉석 확인.
- **게이트 G**: 대표 케이스(텍스트/표/그림 포함) IR 왕복 무손실. 필수 필드 스키마 검증 통과. → `tests/`에 회귀로 고정.

## 검증 — 사용 중 (런타임 가드)
- `from_json`이 받은 dict가 IR 계약을 어기면(필수 필드 없음/타입 불일치) **명확한 에러**로 거부. 침묵 통과 금지.
