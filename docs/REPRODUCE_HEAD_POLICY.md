# 재현 가이드: selection head 결론 보고 (baseline / BC / GRPO)

대상 보고: [`JEV_HEAD_POLICY_CONCLUSION_20261028.md`](JEV_HEAD_POLICY_CONCLUSION_20261028.md)

재현하는 결과:

| 항목 | 기준값 | 재현 단계 |
|---|---|---|
| dev64 결과표 3행 (F1·토큰·도구 호출) | baseline 24.70 / BC 22.50 / GRPO 22.50 | 6 |
| BC 학습 행동 정확도 | 0.998501 (666/667) | 4 |
| teacher 결정 상태 수 | 192문항, 667상태 | 3 |

재현하지 않는 것: 보고의 "H3 대 GRPO 토큰 차이"는 별도 실험(H3 head) 기록입니다. 이 브랜치에는 근거 파일(`lab/reports/decision_h3_role_audit_20261024/correctness_cost.json`)만 남겼습니다.

## 0. 저장된 결과로 표 확인 (GPU 불필요)

```bash
cd lsdrag-code
python scripts/repro_head_policy.py table
```

`lab/reports/decision_shared_validation64_20261023/result.json`에서 보고서 표를 그대로 출력합니다.

## 1. 환경

스크립트가 아래 절대경로를 고정으로 씁니다. 다른 위치를 쓰려면 symlink를 만드세요.

| 경로 | 내용 |
|---|---|
| `/content/local_fast/lsdrag-text-v1` (이하 `ROOT`) | 데이터·모델·실행 결과 |
| `/content/local_fast/lsdrag-v11/embedding-env` | 임베딩 전용 Python venv |
| `/content/local_fast/lsdrag-v11/hf` | 임베딩 모델 HF 캐시 |

**실행 환경** (LLM·head)
- CUDA GPU 1장. bf16 Qwen3-4B 추론과 head 학습을 합니다.
- torch 2.11.0+cu130 (`configs/runtime-requirements.txt` 기준)
- transformers 5.6.2, lm-format-enforcer 0.11.3, jsonschema, numpy

```bash
pip install -e . "transformers==5.6.2" "lm-format-enforcer==0.11.3" jsonschema
```

**임베딩 환경**: `configs/embedding-requirements.txt` (transformers 5.19.0, sentence-transformers 6.1.0)

```bash
python -m venv /content/local_fast/lsdrag-v11/embedding-env
/content/local_fast/lsdrag-v11/embedding-env/bin/pip install torch -r configs/embedding-requirements.txt -e .
```

임베딩 worker는 이 venv에서 `lsdrag.embeddings`를 import하는 별도 프로세스로 뜹니다.

**모델** (revision 고정)

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --cache-dir /content/local_fast/lsdrag-text-v1/hf
hf download google/embeddinggemma-2 --revision 914f7f89142e33e77833254d9c9b90c3cef7303b \
  --cache-dir /content/local_fast/lsdrag-v11/hf   # gated 모델이면 HF 토큰 필요
```

**MuSiQue 데이터와 채점기** ([StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique))

```bash
ROOT=/content/local_fast/lsdrag-text-v1
git clone https://github.com/StonyBrookNLP/musique $ROOT/source/musique_scorer   # metrics/answer.py 사용
(cd $ROOT/source/musique_scorer && bash download_data.sh)
mkdir -p $ROOT/source/musique $ROOT/results
cp $ROOT/source/musique_scorer/data/musique_ans_v1.0_{train,dev}.jsonl $ROOT/source/musique/
sha256sum $ROOT/source/musique/*.jsonl
# train 83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a
# dev   15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b
```

## 2. 공유 코퍼스 구축

```bash
export PYTHONPATH=src:scripts
python scripts/build_decision_shared_corpus.py   # train+dev 문단 중복 제거 → 101,962문단
python scripts/index_decision_shared_corpus.py   # embeddinggemma 768차원 벡터 (상한 2시간)
```

- `documents.jsonl` SHA는 원본 기록과 같아야 합니다: `60561e0e…c6252d`. 입력 데이터만으로 결정됩니다.
- `vectors.npy`는 GPU 수치 차이로 원본 SHA(`e2b7013c…`)와 다를 수 있습니다. 다음 단계부터는 새 SHA로 spec을 다시 만듭니다.

## 3. Teacher 수집 (native 4B, train192)

```bash
python scripts/repro_head_policy.py split     # 고정 split(train192/dev64/test160)을 ROOT/results에 복사
python scripts/repro_head_policy.py teacher   # → lab/registry/repro_head_policy/teacher_spec.json
python scripts/collect_decision_shared_teacher.py --spec lab/registry/repro_head_policy/teacher_spec.json
```

- 출력: `ROOT/results/decision_shared_teacher192_20261023/` (`teacher_states.pt`, `summary.json`)
- 확인: `summary.json`의 `teacher_states`가 667 근처, `errors` 0. 원본은 667, 학습 F1 .2826.
- 시간 상한은 spec의 `max_seconds` 3600입니다.

## 4. 모방 학습 (BC)

```bash
python scripts/fit_decision_shared_initialization.py   # CPU, 수 분
```

- 출력: `ROOT/results/decision_shared_initialization_20261023/head.pt`, `manifest.json`
- 확인: `manifest.json`의 `train_accuracy`가 원본 0.998501에 가까워야 합니다. 학습 상태 기준이라 일반화 지표는 아닙니다.

## 5. GRPO

```bash
python scripts/repro_head_policy.py grpo   # BC head·코퍼스·데이터·코드 SHA를 다시 계산
python scripts/run_decision_shared_grpo.py --manifest lab/registry/repro_head_policy/grpo_manifest.json
```

- 설정은 원본 manifest와 같습니다: seed 20261052, 2회 반복 × 192문항 × 4궤적 = 1,536궤적, lr 1e-4, 시간 상한 3600초.
- 출력: `ROOT/results/decision_shared_grpo192_20261023/grpo_iteration1.pt`, `curves.json`
- 확인: `curves.json`의 `changed_vs_anchor`(BC 대비 argmax가 바뀐 상태 비율)가 작아야 합니다. 원본 결론은 "정책이 거의 바뀌지 않음"입니다.

## 6. dev64 평가

```bash
python scripts/repro_head_policy.py eval   # native_fast / shared_bc / shared_grpo 3조건
python scripts/evaluate_decision_shared_corpus.py --spec lab/registry/repro_head_policy/eval_spec.json
python scripts/repro_head_policy.py table --run decision_shared_validation64_repro
```

마지막 명령은 원본 표와 재현 표를 나란히 출력합니다.

## 재현 판정

GPU 비결정성 때문에 숫자가 정확히 같지는 않습니다. dev64에서 한 문항은 F1 약 1.6%p에 해당합니다. 아래 결론이 유지되는지로 판단합니다.

1. BC head의 도구 호출 수가 baseline과 비슷하다(모방 성공).
2. GRPO의 F1이 BC와 같거나 거의 같다(정확도 정책 없음).
3. GRPO의 토큰 변화가 몇 % 수준이다.
4. 두 head 모두 baseline F1보다 높지 않다.

## 주의

- 스크립트는 출력 폴더가 이미 있으면 멈춥니다(`exist_ok=False`). 다시 돌리려면 해당 `ROOT/results/<run>`을 옮기거나 지우세요.
- 원본 기록(`lab/registry/decision_shared_*_20261023/`)은 수정하지 않습니다. 재현 spec은 `lab/registry/repro_head_policy/`에만 씁니다.
- `ROOT/decision_policy.lock`으로 GPU 단계가 직렬화됩니다. 단계를 동시에 실행하지 마세요.
- 이 가이드의 전체 GPU 경로는 v2.0 정리 이후 다시 실행해 보지 않았습니다. 0단계 표 출력과 spec 생성만 확인했습니다.

## 파일 구성

| 단계 | 스크립트 | 원본 기록 |
|---|---|---|
| 코퍼스 | `scripts/build_decision_shared_corpus.py`, `scripts/index_decision_shared_corpus.py` | `lab/registry/decision_shared_corpus_20261023/` |
| teacher | `scripts/collect_decision_shared_teacher.py` | `lab/registry/decision_shared_teacher192_20261023/spec.json`, `lab/reports/decision_shared_teacher192_20261023/audit.json` |
| BC | `scripts/fit_decision_shared_initialization.py` | 정확도 .998501, head SHA `0023dd45…` |
| GRPO | `scripts/run_decision_shared_grpo.py` | `lab/registry/decision_shared_grpo192_20261023/manifest.json` |
| 평가 | `scripts/evaluate_decision_shared_corpus.py` | `lab/registry/decision_shared_outcomes192_20261023/validation_spec.json`, `lab/reports/decision_shared_validation64_20261023/` |
| split | — | `lab/registry/decision_state_policy_iteration_20261023/manifest.json` |
| 보조 | `scripts/repro_head_policy.py` | spec 재생성, 표 출력 |

핵심 구현: `src/lsdrag/decision_policy.py` (`PolicyHead`, `rollout`, `distribution`), `src/lsdrag/decision_grpo.py` (`group_advantages`), `src/lsdrag/decision_policy_recovery.py` (`RecoveryPolicyReader`).
