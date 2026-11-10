# Gate 누적 기록

| 단계 | 기준 | 측정 | 판정 | 근거 | UTC |
|---|---|---|---|---|---|
| S0 assets | 원본/코퍼스/벡터/lexical hash 일치100% | 100% | PASS | RESTORE_VERIFIED.json; specs/v22/preflight.json | 2026-11-10T04:48:55.987999+00:00 |
| S0 tests | 실패0 | 18 passed | PASS | runtime pytest tests 실행 | 2026-11-10T04:48:55.987999+00:00 |
| S0 answer cap | 모든-alias 차단 qid0 | cap60, 0/576 | PASS | specs/v22/preflight.json | 2026-11-10T04:48:55.987999+00:00 |
| S0 KD split | qid overlap0 | fit459/val117, qid overlap0; family 공유59 val qid | PASS(qid만) | specs/v22/kd_split.json; preflight.json | 2026-11-10T04:48:55.987999+00:00 |
| S0 train16 smoke | 구현 오류0 | 미측정 | PENDING | results/v22_legacy_smoke16 | 2026-11-10T04:48:55.987999+00:00 |
| S0 legacy dev192 | 기존 수치와 차이 조사 완료 | 미측정 | PENDING | specs/v22/legacy_dev192.json | 2026-11-10T04:48:55.987999+00:00 |
| S1~S5 | PLAN.md 기준 | 미측정 | NOT STARTED | PLAN.md | 2026-11-10T04:48:55.987999+00:00 |
| S0 train16 smoke | 구현 오류0 | 16/16 완료, 오류0 | PASS(구현) | results/v22_legacy_smoke16/summary.json | 2026-11-10T04:50:48.815327+00:00 |
| 구현 CPU | pointer 가변 K/마스크/gradient/replay | 7 passed | PASS(기능) | tests/test_chunk_pointer.py | 2026-11-10T04:57:27.900737+00:00 |
| 구현 CPU | 정답 우선/단계 credit/부호 있는 A_t | 3 passed | PASS(함수) | tests/test_v22_reward.py | 2026-11-10T04:57:27.900737+00:00 |
| 구현 CPU | 실제 tokenizer 후보 span 매핑 | 2/2 exact span | PASS(작은 smoke) | specs/v22/span_smoke.json | 2026-11-10T04:57:27.900737+00:00 |
| S0 legacy dev192 | 과거192문항 trajectory/answer 및 주요 집계 재현 | 192/192 일치、F1/EM/calls/tokens 차이0、errors0 | PASS | codex/S0_LEGACY_AUDIT.json | 2026-11-10T04:58:55.295428+00:00 |
| S1 smoke16 | errors0/span 및 ID 매핑/1-step KD 손실 감소 | 16episodes errors0; read7; dispatch7/7; loss2.399941→1.802975 | PASS(기능) | codex/S1_SMOKE.json; results/v22_teacher576_20261110 | 2026-11-10T05:04:28.320286+00:00 |
| 구현 CPU replay | FP32 cached vs fullprefix 단위logp 오차<=1e-4/실제 backbone gradient | 소형 Qwen 2 tests PASS | PASS(작은 FP32) | tests/test_v22_replay.py; 실제4B BF16미측정 | 2026-11-10T05:12:05.648317+00:00 |
| 구현 CPU | 전체 회귀 및 optimizer 재개 다음 step 일치 | 43 passed | PASS(기능만) | tests/; 실제 full-FT 미실행 | 2026-11-10T05:23:53.412026+00:00 |
| S1 teacher audit | 576 QA/모든 상태 후보span·라벨 provenance 일치 | 576 QA / 2065states 일치, invalid read1 제외 | PASS | codex/S1_TEACHER_AUDIT.json | 2026-11-10T05:32:59.009866+00:00 |
| S1 coverage | fit read>=100; val hop별>=10 | fit280; 2hop25/3hop26/4hop27 | PASS | codex/S1_TEACHER_AUDIT.json | 2026-11-10T05:32:59.009866+00:00 |
