현재 단계: S1 pointer 품질 gate 미통과 — 후속 실험 중단·보고/백업 마무리
마지막으로 통과한 gate: teacher576 전체 감사 및 coverage fit280/val25·26·27
실행 중 PID: 없음 (teacher52782/KD75120 종료 확인)
결과 경로: /content/local_fast/lsdrag-text-v1/results/v22_kd_20261110; codex/REPORT_S1.md

# Codex 실행 기록

사용자 기준 폴더: `/content/drive/MyDrive/lsdrag/codex/`
작업/Git: `/content/local_fast/lsdrag-code-v2.2`, branch `jev_v2.2`.

- [S1 중단 판정 보고서](REPORT_S1.md) / [수치 원본](REPORT_S1.json)
- [현재 상태](CURRENT.md) / [누적 gate](GATES.md) / [실행 이력](SESSION_LOG.md)
- [사전 계획 개정6](PLAN.md) / [결과 비교 템플릿](COMPARISON_TEMPLATE.md)
- [Teacher 감사](S1_TEACHER_AUDIT.json) / [Drive teacher 백업](S1_BACKUP.json)
- [HF 데이터 및 원격 검증](HF_DATASET.json)

HF: https://huggingface.co/datasets/sungkwang2/lsdrag-v22-generated (private).
Teacher576와 KD100epoch 완료. Pointer 내부검증 F1 .6979798 < teacher .7141414로 품질 gate 실패.
S2 자율QA 비교, S3 보상pilot, full-FT smoke, H-RL/N-RL GRPO, test320 미실행.
새 지시 없이 후속 실험 자동실행하지 않음. Claude 파일은 읽기 전용이며 OBS02까지 반영했다.
