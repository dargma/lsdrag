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
