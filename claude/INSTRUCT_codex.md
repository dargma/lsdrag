# Codex에게 — 관찰자(Claude)와 함께 실행할 때 지킬 것

작성: Claude, 2026-11-10. 사용자 요청으로 저장.

1. 지금 `claude/OBS_01_run_start.md`를 읽고 항목별로 반영.
2. 첫 GPU 실행 전에 `codex/PLAN.md`(개정 2), `codex/README.md`, `codex/RESPONSE_02.md`를 커밋·push. 기준을 결과 전에 고정했다는 증거임. 이후 기준을 바꾸면 새 커밋에 이유를 적음.
3. "전부 완료" 목표보다 PLAN의 gate가 우선함. gate에서 걸리면 멈추고, 그때까지의 결과로 보고서와 백업을 끝내는 것이 완료임. 기준을 낮추거나 온도·계수를 더 탐색해 통과시키지 않음. PLAN에 "사용자에게 확인"이라고 적힌 지점에서는 멈추고 물음.
4. 단계가 바뀔 때마다 `codex/README.md` 맨 위 네 줄 갱신: 현재 단계 / 마지막으로 통과한 gate / 실행 중 PID / 결과 경로.
5. gate의 실제 숫자를 `codex/GATES.md` 한 파일에 누적. 한 gate당 한 줄: 단계, 기준, 측정값, 통과 여부, 근거 파일 경로, UTC 시각. 안 잰 것은 "미측정".
6. Claude 피드백은 `claude/OBS_NN_*.md`로 올라옴. 새 파일이 있으면 다음 단계 전에 읽고 `codex/RESPONSE_OBS_NN.md`에 항목별 반영/거절/보류와 한 줄 이유를 적음. BLOCK은 해소 전에 다음 단계로 가지 않음.
7. `claude/`는 읽기만. Claude는 `codex/`와 코드를 수정하지 않음.
