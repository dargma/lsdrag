# 07 — agent (단계 3: 검색 — agent loop 연결)

## 목적
A-RAG agent loop에 tool registry(06)를 연결하고, `image_read`용 context 확장을 적용해 전체 질의 파이프라인을 완성.

## 선행 / 산출물
- 선행: 06(tool 5종), 01_FACTS(loop·context).
- 산출물: `src/agent/` — vendored A-RAG + `AgentContext` 이미지 추적 확장 + 5 tool 연결 진입점.

## 설계 결정 — A-RAG 가지치기 (필수, 불필요 요소 삭제)
- **"우리 런타임 경로가 import하는가?"** 기준. 아니면 가져오지 말거나 삭제.
  명백히 불필요: `scripts/eval.py`, `scripts/batch_runner.py`, 평가/벤치 코드, 미사용 tool, 예제/데모.
- vendor에 남길 최소 집합: `core/{context,llm,config}.py`, `tools/{base,registry,semantic_search,keyword_search,read_chunk}.py`, `agent/base.py`. 이 외는 실제 의존 확인 시에만.
- `src/agent/vendor/SOURCE.md`에 **커밋(`a44de6b`)·가져온 파일·삭제한 파일·이유·라이선스(MIT)** 기록.

## 설계 결정 — 연결
- `AgentContext` 확장은 **서브클래스 우선**(vendor 직접 수정 회피). 청크 추적 패턴을 이미지에 복제(사실 4: `read_image_ids`).
- Reader = GPT-4.1 mini(멀티모달), config의 reader 블록(엔드포인트·모델·키 env). 이미지 직접 처리(별도 VLM 없음).
- loop 파라미터(`max_loops` 등)도 config. **loop 자체는 건드리지 않는다**(무한루프 방지 이미 존재, 사실 5).

## 단독 실행 (단계 분리 증명)
```bash
python -m src.agent --query "What is the SCTLR_EL1 register?" --config config.yaml
```

## 검증 — 작업 중 (Inline + 게이트)
- Inline: mock tool들로 1턴 실행 → tool 호출 → 답변 생성까지 흐르는지. 같은 이미지 재호출 시 context가 막는지(end-to-end 미니).
- **게이트 G**: 실제 5 tool 연결 상태에서 (a) 텍스트 질의 (b) 구조 질의(page/figure) (c) 이미지 동반 질의 각각 loop 완주·답변. context에 이미지 추적 기록. → 회귀 고정.

## 검증 — 사용 중 (런타임 가드)
- VLM(Reader) 실패 → 텍스트 컨텍스트만으로 폴백 + 로그.
- tool 예외 → agent에 에러 문자열 반환(registry가 이미 처리, 사실 2). 크래시 금지.
