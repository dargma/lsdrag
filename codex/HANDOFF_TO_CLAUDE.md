# Claude 실행 인계 — 2026-11-10T05:40:10.650140+00:00

사용자가 “claude가 이어서 실험 진행할거야”라고 지정했다. 이후 실험 실행 담당은 Claude이며 Codex는 새 worker를 시작하지 않는다. 이 지정은 이전 실패 결과를 통과로 바꾸는 것이 아니다. 다음 실험의 변경점·근거·gate를 먼저 기록하고 기존 결과는 보존한다.

## 먼저 읽을 기록

`codex/REPORT_S1.md`, `codex/CURRENT.md`, `codex/GATES.md`, `codex/PLAN.md`(개정6), `codex/RESPONSE_OBS_02.md`, `claude/INSTRUCT_codex.md`, `claude/OBS_02_plan_rev5.md`.
인계 직전 Git 기준30579603a13be340718b7019b7a9e7938c9cb02b. 실제 작업트리는 `/content/local_fast/lsdrag-code-v2.2`, branch `jev_v2.2`. `/content/local_fast/lsdrag-code`의 v2.1 및 Drive의 옛v1.1 memory와 혼동하지 말 것. 사용자 표시/문서 복사 위치는 `/content/drive/MyDrive/lsdrag/codex/`.

## 확인된 결과

- S0 legacy dev192 답·trajectory192/192 재현, CPU43검사 통과.
- Teacher576 완료,2065학습 상태,358valid read. 모든 후보span/라벨 감사 통과.
- Coverage fitread280, 내부검증2/3/4hop25/26/27로 통과.
- 중복ID 생성실패QA1 보존, pointer라벨1 제외. source별 gold296/fallback62.
- KD100epoch 완료, validation 합산CE 최소 epoch index14 선택.
- **S1 품질 FAIL**: gold-present 내부검증66qid pointer supportF1 .69797979798 < teacher .71414141414. 16승17패33동률; fallback12상태 ordered exact .5.
- S2 공통계약N0/H0-P, 조건부H0-G, S3 rewardpilot, 실제fullFTsmoke, H-RL/N-RL GRPO, test320 모두 미실행. GPU유휴는 이 중단 결과다.
- 종료 확인한 기존 PID52782(teacher),75120(KD). 새 worker가 있는지는 재개 직전에 다시 확인할 것.

## 재사용 경로

공통 ROOT `/content/local_fast/lsdrag-text-v1`:

| 항목 | ROOT 기준 경로 |
|---|---|
| Python | `runtime/bin/python` |
| teacher 원본/라벨 | `results/v22_teacher576_20261110/` |
| KD checkpoint/전체검증·curve | `results/v22_kd_20261110/` |
| frozen fit/validation | `specs/v22/kd_split.json` |
| 전체 teacher 감사 | `specs/v22/teacher_audit.json` |
| teacher 수집 소스 | `source_snapshots/v22_s1/` |
| 감사/KD 실행 소스 | `source_snapshots/v22_s1_audit_6a51973/` |
| corpus | `corpora/musique_shared_20261023/` |
| 모델 경로 | 작업트리 `configs/experiment.json`의 model |

KD head SHA256 `86b163fa2106a98bf66fb8ed3ab6027fab2c53df6a397923b2b077bce7d06781`.
Teacher tensor SHA256 `08fd04bd0158064fc84622c407c32a57a775c4f6ce1bebfe12f573443b9591fb`.
HF 비공개 https://huggingface.co/datasets/sungkwang2/lsdrag-v22-generated,
revision `0afa9ef4eb6ce85d0b263d7fde12969f5526ec67`. episodes459/117,labels1646/419,
원본tensor·split·스키마·소스·KD실패판정 포함.62파일 원격해시 및두config 로딩 검증.
Drive원본/최종백업 위치와SHA는 `S1_BACKUP.json`, `FINAL_BACKUP.json`. 모델weights는Git/HF dataset에 포함하지 않았다.

## 후속 실행 전 알아둘 한계

1. 기존 run 출력 폴더를 덮어쓰지 말 것. 새 spec/output/source snapshot을 고정하고 실제 사용한 commit/hash를 기록.
2. 현재 gate는 미달이다. 결과를 본 후 gate를 바꾸면 새 실험/변경으로 명시하며 기존FAIL을 소급 수정하지 않는다. 추가학습으로 자동통과시키는 재시도는 기존계획에 포함되지 않았다.
3. `run_fullft_grpo.py --manifest ... --collect-only/--update-only` 경로는 구현했지만 실제4B GPU 검증 전이다. CPU 통과를 본학습 준비완료로 해석하지 말 것.
4. 특히 smoke의 post-update greedy16/분포변화·최종KL·비용항비중·round판정 연결, on-policy 모델/라벨 provenance, optimizer/RNG재개 계약을 실행 전에 검토해야 한다. pilot 수집은24qid, main은192qid로 검사하므로 별도 smoke16×8 수집 계약도 정합성을 확인할 것.
5. HF export: `scripts/export_v22_teacher_hf.py`; 보고 재생성: `scripts/report_v22_s1.py`. 최신 HF README의 KD실패 추가 문구와 최종Drive registry 링크는 별도 마무리 단계에서 추가했다.
6. Fit459와 내부검증117은 qid-disjoint지만59validation qid가 atomic family를 공유. 독립holdout으로 부르지 않는다. dev/test를 튜닝에 쓰지 않는다.
7. 기본 reward cc=.15/ct=.10. Offline12는진단만. OBS02에 따른 bounded smoke 조건/사용자확인gate 유지. H0-G는 필수비교가 아니며 공통N0보다H0-P EM또는F1이 낮을 때만진단.

후속 생성 데이터도 사용자 요청에 따라 HF에 단계별 업로드하고 immutable revision·split·오류·gate결과를 함께 보존할 것. 인증값을문서나Git에 기록하지 말 것.
