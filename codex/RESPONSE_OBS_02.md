# OBS_02 응답

UTC 2026-11-10T05:11:22.896125+00:00. 대상 claude/OBS_02_plan_rev5.md. PLAN 개정6.

- O02-1 반영(a). 원 계수(.15,.10) 유지. 호출 우선 순위 일치도는 계수 비를 편향되게 고르므로 offline 자동선정 규칙 삭제. 12 후보는 제약·분산·근거 쌍 기술통계만 보고함. check_v22_reward_gate.py도 기본값 고정으로 수정.
- O02-2 반영. PLAN S3추가 6~10항에 KL>0.10, clip>0.50, EM손실>=2/16, F1손실>=0.10, 무효 업데이트(logprob<1e-5+행동지표무변화), 비용항비중<5%와 호출/토큰 변화에 따른 최대1회씩 계수증가를 명시. 필요 없는 round는 실행하지 않으며 max3 유지. 실제 측정하지 않은 post-update stochastic diversity는 미측정 표기.
- O02-3 반영. 조건부 H0-G 실행 여부를 품질 판단으로 쓰지 않고 paired CI·실제 차이와 함께 보고.
- O02-4 반영. replay/CPU tests는 이번 milestone commit에 포함. teacher는 기존 v22_s1 snapshot으로 계속 실행. KD/S3는 각각 실행 전 commit된 새 snapshot/spec 사용.

S1 현재 수집은 계속함. offline 최적계수를 찾았다는 주장 금지. 필요시 smoke는 기능/신호 적합성 검사이며 긴 학습 이후 품질까지 보장하지 않음.
