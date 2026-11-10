# Claude 기록 검토 — 2026-11-10

읽은 파일: claude/README.md, claude/PROTOCOL.md, claude/RESEARCH_BRIEF_select_heads_write_tokens_20261110.md.

- 반영: codex/만 작성하고 claude/는 읽기만 함. 리뷰가 도착하면 항목별 반영/거절/보류와 근거를 RESPONSE_NN.md에 기록함.
- 반영: 목표는 iteration 감소·생성 감소·정확도 개선이며 최소 PoC, smoke 선행.
- 반영: 동적 pointer는 후보 내용 표현에 shared scoring을 적용함.
- 반영: constrained native를 대조군으로 유지. 이미 ID enum 제약이 있으므로 전사 오류 제거가 고유 이점이라고 주장하지 않음.
- 반영: group std 정규화를 피하고 head 전체/LLM 전체 gradient와 rollout 다양성을 검증함.
- 변경 제안: 복수 선택은 독립 sigmoid 대신 순차 pointer+STOP을 우선함. 1~8개 제약과 학습/rollout의 정확한 logprob를 정의하기 쉬움. 성능 우위가 입증된 선택은 아님.
- 보류: support gold로 직접 pointer/FINAL 라벨 만들기, 결정별 별도 advantage, Q-head, AtomMem 확장. 이번 사용자 요청은 우선 teacher KD 후 공통 outcome reward의 GRPO8 대조 실험임.
- 정정: 한 번의 forward로 생성 전체까지 끝나는 구조는 아님. 같은 최초 prefill을 공유하지만 검색 인자·답 생성에는 후속 autoregressive forward가 필요함.
- 정정: 원문 브리프의 과거 PoC 수치와 최신 v2.1 보고 수치는 섞지 않음. 최신 frozen split은 test만 160/80/80이며 모든 split 균등이 아님.
- 미해결: loss normalization/temperature/label coverage/학습 예산은 실행 전 동결 필요. 본 계획은 실행 승인이 아님.
