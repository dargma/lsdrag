# Claude 실행 기록 (v2.2 인수 이후)

2026-11-10T05:40Z부터 사용자 지시("너가 가져와서 실험 이어서 진행해")로 Claude가 실행을 맡음. Codex가 남긴 `codex/`는 수정하지 않고, 이후 상태·gate·기록은 이 파일에 누적함. 계획 기준은 `codex/PLAN.md` 개정 6.

## 현재 상태

- 단계: S2 (N0 / H0-P dev192 평가) 준비
- 마지막 gate: S1 pointer 품질 FAIL → 사용자 지시로 진행 (아래)
- 실행 중 PID: 없음
- 결과 경로: `/content/local_fast/lsdrag-text-v1/results/`

## 인수 시점 판단

S1 pointer 품질 gate는 **FAIL로 유지**함 (pointer 0.6980 < teacher 0.7141, 내부검증 gold-present 66문항). 통과로 바꾸지 않음. 사용자가 실패를 안 상태에서 계속 진행을 지시했으므로, 이후 결과는 모두 "S1 품질 gate 미통과 상태에서 사용자 지시로 진행"이라는 조건을 달고 보고함.

Claude가 `results/v22_kd_20261110/summary.json`의 66개 상태를 다시 계산한 값:

| 항목 | pointer | teacher |
|---|---:|---:|
| 첫 선택이 gold | 50/66 | 51/66 |
| gold를 하나라도 선택 | 51/66 | 61/66 |
| 평균 선택 수 | 1.03 | 1.68 (gold 라벨 1.26) |
| precision | 0.763 | 0.702 |
| recall | 0.667 | 0.818 |
| 2개 이상 선택한 상태 | 1/66 | – (gold가 2개 이상인 상태 17/66) |

- 해석: 순위는 teacher와 같은 수준이고, 차이는 pointer가 거의 항상 1개만 고르고 멈추는 데서 나옴.
- 차이 −0.016은 66문항 중 약 1문항 크기이며 Codex 보고의 paired bootstrap 95% 구간은 [−0.110, +0.078]임. gate가 점추정 부등호여서 걸렸음.
- 위험: 자율 실행에서 한 번에 1개만 읽으면 근거를 덜 읽거나 read 호출이 늘 수 있음. S2의 H0-P dev192에서 read당 선택 수, read 호출 수, read_support_recall로 확인함.
- KD 재학습·선택 수 보정은 하지 않음. 내부검증을 본 뒤의 조정이 되기 때문임. 필요해지면 별도 항목으로 사용자에게 제안함.

## 진행 규칙

- PLAN 개정 6의 나머지 gate와 "사용자에게 확인" 지점은 그대로 지킴. dev/test로 튜닝하지 않음.
- 실행 소스는 커밋된 해시의 스냅샷(`source_snapshots/`)만 사용함.
- test320은 주 checkpoint 동결 전에 열지 않음.

## Gate 기록 (인수 이후)

| 단계 | 기준 | 측정 | 판정 | 근거 | UTC |
|---|---|---|---|---|---|
| S1 pointer 품질 | pointer support F1 ≥ teacher | 0.6980 < 0.7141 (66문항) | FAIL, 사용자 지시로 진행 | `codex/REPORT_S1.md` | 05:40 |

## 로그
