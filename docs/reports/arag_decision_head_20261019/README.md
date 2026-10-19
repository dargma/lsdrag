# A-RAG decision head 실험 결과 보고서

- [HTML 보고서](index.html)
- [브라우저 미리보기](https://htmlpreview.github.io/?https://github.com/dargma/lsdrag/blob/jev_v1.1/docs/reports/arag_decision_head_20261019/index.html)
- [검증된 소규모 근거 JSON](evidence.json) / [검증 기록·산출물 해시](validation.json)
- [대표 문항 Head 원본](raw_head_generate_2hop__391857_162018.json), [Native 원본](raw_native_2hop__391857_162018.json)

2026-10-17 실험을 2026-10-19에 정리했다. 원본 200개 episode 해시와 두 archive 해시를 확인하고 집계를 재계산했다. 새로운 모델 실행·학습·평가 없이 보고서만 작성했다. 실험 HOLD 유지.

보고서에는 배경, 환경, 실제 teacher 라벨 구성, action/read head 및 KV cache, tool별 입력과 호출 현황, 100문항 정량 결과, read 선택 실패, 성공/실패 사례, 한 문항의 전체 raw step, 후속 연구 제안이 포함된다. HTML의 검색·승패 필터는 전체 100문항 중 표시만 바꾼다.

정량 결과는 공유 GPU에서 관측한 소규모 평가다. 품질 개선·비열등성·head 단독 속도 향상은 입증하지 않았다. Native는 이 저장소의 동일 Qwen 기반 A-RAG 방식 비교군이며 공식 원 논문 전체 재현이 아니다.

재생성: 저장소 루트에서 `python scripts/reporting/build_arag_decision_report.py`. 표·사례는 이 폴더의 evidence.json과 명시적인 보고 prose를 사용한다. 모델과 원시 데이터 전체는 필요하지 않다.
