# OBS_01 응답

UTC 2026-11-10T04:48:55.987999+00:00

- O01-1 반영. GPU smoke PID38568이 계획 commit 전에 이미 기동됨. 지적을 읽은 뒤 SIGSTOP으로 멈췄고 실제 경과를 기록함. 이 smoke는 사전 동결 증거가 없는 구현/복원 진단으로만 사용. 이 커밋 이후 본 평가와 수집을 진행하며 기준 변경은 별도 기록.
- O01-2 반영. 사용자 최신 지시대로 gate 실패 시 다음 단계로 가지 않고 해당 시점 보고·Drive/Git 백업을 완료함. 사용자 확인 gate는 질문하고 응답 전 S4 실행하지 않음. 본 실험 미실행을 완료로 위장하지 않고 중단 범위를 명시함.
- O01-3 반영. README 첫 네 줄에 단계·마지막 gate·PID·결과 경로 기록.
- O01-4 반영. GATES.md 누적. legacy dev192 수치 비교는 아직 미측정. train16 smoke를 dev192 재현으로 대체하지 않음.

S0 발견: internal validation117 중59 qid가 fit과 atomic family를 공유함. qid 분리는 유지되지만 family-disjoint 일반화라고 주장하지 않음. 답 cap60에서 모든 정규형 alias가 막히는 train qid0.
