# REVIEW_01 — codex/RESTORE_VERIFIED.json

- 대상: `codex/RESTORE_VERIFIED.json` (미커밋, sha256 `5cdf1717b40a`), 읽은 시각 2026-11-10T04:24Z
- 이 시각에 `codex/`에는 이 파일 하나뿐임. `PLAN.md`, `COMPARISON_TEMPLATE.md`는 없어서 검토하지 못함.
- Claude가 직접 확인한 것: 파일 존재와 크기만. 해시는 다시 계산하지 않음.

## 확인된 것
- Qwen3-4B-Instruct-2507 가중치 3개(약 8.0GB)와 embeddinggemma-2(1.49GB) 존재.
- 코퍼스 `musique_shared_20261023`: documents.jsonl, lexical.sqlite, vectors.npy 존재. vectors.npy 크기 313,227,392바이트 = 101,962 × 768 × 4 + 128 헤더로 기록된 shape과 일치.
- MuSiQue 원본 train/dev jsonl 존재. split 수치는 `configs/split_mixhop.json` 설명(576 / 192 / 320)과 일치.

## 의견

**R01-1 [ASK] 복원한 head가 어느 실험 것인지**
- 문제: 복원된 head 3개는 `decision_shared_initialization_20261023`과 `decision_shared_grpo192_kl0`로, mixed-hop 전환(2026-11-03) 이전의 2hop·dev64 시기 산출물임.
- 근거: `HANDOVER.md`의 mixed-hop 실험(B1/C1) head와 A1/B1 가중치는 Drive `lsdrag-backups/20261104/`에 있고 이번 복원 목록에 없음.
- 제안: 1차 KD의 초기값·비교 기준으로 어느 head를 쓸지 PLAN에 명시. 새 구조라 처음부터 학습한다면 이 3개는 필요 없으니 그렇게 적으면 됨.

**R01-2 [FIX] 실행 환경이 없음**
- 문제: `/content/local_fast/lsdrag-text-v1/`에 venv가 없음. 지금은 smoke도 못 돌림.
- 근거: 디렉터리 목록(corpora, hf, logs, restore_20261110, results, source). Codex도 04:12Z에 "Python 환경 미복원"이라고 보고함.
- 제안: PLAN의 0번 단계로 "환경 복원 + 기존 테스트 18개 통과 + 기존 baseline 2~4문항 smoke"를 넣기.

**R01-3 [ASK] teacher·이전 rollout 재사용 여부**
- 문제: `teacher192`와 mixed-hop 결과(episodes)가 복원되지 않음.
- 근거: `results/` 아래에 위 두 폴더뿐임.
- 제안: 1차 라벨을 새로 수집할지, 기존 mixed-hop episodes에서 뽑을지 PLAN에 명시. 기존 episodes를 쓰면 GPU 없이 라벨 일부를 만들 수 있어 PoC가 빨라짐.

**R01-4 [NOTE] test320은 hop 비율이 160/80/80**
- train/validation만 균등함. 최종 수치를 hop별로 나눠 보고하거나 가중 평균임을 표시해야 함. Codex도 이미 인지함.
