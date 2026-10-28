# lsdrag v2.0: 도구 선택 head 학습 (BC → GRPO)

A-RAG 에이전트(Qwen3-4B-Instruct-2507, MuSiQue 2hop)에서 "다음에 쓸 도구"를 작은 head(약 34만 파라미터)가 고르도록 학습합니다.
1. native 에이전트를 따라 하는 모방 학습(BC)으로 head를 만듭니다.
2. 그 head를 GRPO로 더 학습합니다.
3. 결과를 LLM이 직접 도구를 고르는 baseline과 비교합니다.

- 결론 보고: [`docs/JEV_HEAD_POLICY_CONCLUSION_20261028.md`](docs/JEV_HEAD_POLICY_CONCLUSION_20261028.md)
- 상세 재현 가이드(해시, 확인 항목, 주의 사항): [`docs/REPRODUCE_HEAD_POLICY.md`](docs/REPRODUCE_HEAD_POLICY.md)

| 조건 (dev64) | F1 (%) | 생성 토큰 | 입력 토큰 | 도구 호출 |
|---|---:|---:|---:|---:|
| baseline (native) | 24.70 | 97.4 | 4,691 | 2.38 |
| 모방 (BC head) | 22.50 | 101.6 | 4,749 | 2.39 |
| GRPO 적용 | 22.50 | 98.4 | 4,625 | 2.33 |

다른 실험과 전체 이력은 `jev_v1.1` 브랜치에 있습니다.

## 전체 흐름

```
MuSiQue ─▶ ① 코퍼스 ─▶ ② teacher 수집 ─▶ ③ BC head ─▶ ④ GRPO ─▶ ⑤ dev64 평가
           문단+벡터     native 4B 결정 상태   모방 학습      head 강화학습   결과표
```

| 단계 | 명령 | 결과물 | 시간(참고) |
|---|---|---|---|
| ① 코퍼스 | `build_decision_shared_corpus.py`, `index_decision_shared_corpus.py` | 101,962문단, 768차원 벡터 | CPU 수 분 + GPU 30분 안팎 |
| ② teacher | `collect_decision_shared_teacher.py` | train192에서 결정 상태 667개 | GPU 1시간 이내 |
| ③ BC | `fit_decision_shared_initialization.py` | `head.pt` | CPU 수 분 |
| ④ GRPO | `run_decision_shared_grpo.py` | `grpo_iteration1.pt` | GPU 약 1시간 |
| ⑤ 평가 | `evaluate_decision_shared_corpus.py` | dev64 결과표 | GPU 30분 이내 |

모든 출력은 `/content/local_fast/lsdrag-text-v1`(이하 `ROOT`) 아래에 생깁니다. 경로는 스크립트에 고정돼 있습니다.

## 0. 설치 없이 결과표 확인

```bash
git clone -b jev_v2.0 https://github.com/dargma/lsdrag lsdrag-code && cd lsdrag-code
python scripts/repro_head_policy.py table
```

## 1. 환경과 모델, 데이터

```bash
ROOT=/content/local_fast/lsdrag-text-v1
mkdir -p $ROOT/hf $ROOT/source/musique $ROOT/results $ROOT/logs /content/local_fast/lsdrag-v11/hf

# 실행 환경: 시스템 torch(2.11, CUDA)를 그대로 쓰는 venv
uv venv --system-site-packages -p $(which python3) $ROOT/runtime
uv pip install -p $ROOT/runtime/bin/python "transformers==5.6.2" "lm-format-enforcer==0.11.3" jsonschema -e .

# 임베딩 환경: --no-deps 필수(없으면 torch가 따로 깔려 import 실패)
uv venv --system-site-packages -p $(which python3) /content/local_fast/lsdrag-v11/embedding-env
uv pip install --no-deps -p /content/local_fast/lsdrag-v11/embedding-env/bin/python \
  "transformers==5.19.0" "sentence-transformers==6.1.0" -e .

# 모델 (revision 고정)
hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --cache-dir $ROOT/hf
hf download google/embeddinggemma-2 --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --cache-dir /content/local_fast/lsdrag-v11/hf

# MuSiQue 데이터와 공식 채점기
git clone https://github.com/StonyBrookNLP/musique $ROOT/source/musique_scorer
gdown --id 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h --output $ROOT/musique_v1.0.zip
python -c "import zipfile;z=zipfile.ZipFile('$ROOT/musique_v1.0.zip');[open('$ROOT/source/musique/'+n.split('/')[-1],'wb').write(z.read(n)) for n in z.namelist() if n.endswith(('ans_v1.0_train.jsonl','ans_v1.0_dev.jsonl')) and 'MACOSX' not in n]"
sha256sum $ROOT/source/musique/*.jsonl   # train 83a75b1e…, dev 15fa6379…
```

이후 명령은 모두 저장소 루트에서 아래 설정으로 실행합니다.

```bash
export PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2
PY=/content/local_fast/lsdrag-text-v1/runtime/bin/python
```

## 2. 데이터셋 만들기 (① 코퍼스, ② teacher)

```bash
$PY scripts/build_decision_shared_corpus.py          # documents.jsonl, SHA 60561e0e…와 같아야 함
$PY scripts/index_decision_shared_corpus.py          # vectors.npy
git checkout lab/registry/decision_shared_corpus_20261023/corpus_manifest.json   # 원본 기록 되돌리기

$PY scripts/repro_head_policy.py split               # train192 / dev64 / test160 split 배치
$PY scripts/repro_head_policy.py teacher             # 새 해시로 teacher spec 생성
$PY scripts/collect_decision_shared_teacher.py --spec lab/registry/repro_head_policy/teacher_spec.json
```

확인: `ROOT/results/decision_shared_teacher192_20261023/summary.json`의 `teacher_states`가 667 근처여야 합니다.

## 3. head 학습 (③ BC → ④ GRPO)

```bash
$PY scripts/fit_decision_shared_initialization.py    # BC. manifest.json의 train_accuracy가 0.9985 근처

$PY scripts/repro_head_policy.py grpo                 # GRPO manifest (KL 0.02, seed 20261052)
$PY scripts/run_decision_shared_grpo.py --manifest lab/registry/repro_head_policy/decision_shared_grpo192_20261023_manifest.json
```

## 4. 평가 (⑤)

```bash
$PY scripts/repro_head_policy.py eval                 # native / BC / GRPO 3조건
$PY scripts/evaluate_decision_shared_corpus.py --spec lab/registry/repro_head_policy/decision_shared_validation64_repro_spec.json
$PY scripts/repro_head_policy.py table --run decision_shared_validation64_repro   # 원본 표와 나란히 출력
```

GPU 비결정성 때문에 숫자가 똑같지는 않습니다. dev64에서 1문항은 F1 약 1.6%p입니다. 판정 기준은 재현 가이드에 있습니다.

## 한 번에 실행

```bash
bash scripts/repro_pipeline.sh
```

①~⑤를 순서대로 실행하고, 이미 끝난 단계는 건너뜁니다. KL 0 변형 GRPO까지 포함합니다. 로그는 `ROOT/logs/`에 남습니다.

## 학습된 가중치

BC·GRPO head는 HF `sungkwang2/lsdrag-v2-head-policy`(private)에 있습니다. 받아서 `ROOT/results/decision_shared_initialization_20261023/head.pt` 등에 두면 ③④를 건너뛸 수 있습니다.

## 변형 실험

| 실험 | 명령 |
|---|---|
| GRPO에서 head KL 계수 변경 | `repro_head_policy.py grpo --name <run> --kl 0` 후 `run_decision_shared_grpo.py --manifest …/<run>_manifest.json` |
| LLM에 LoRA를 붙여 head와 함께 GRPO | `run_lora_head_grpo.py --name <run>`. 평가는 `LORA_PATH=ROOT/results/<run>/lora_iteration0.pt $PY scripts/eval_lora_head.py --spec <spec>` |
| LoRA 학습이 끝나면 자동 평가 | `bash scripts/lora_followup_eval.sh <run>`. LoRA+head와, 같은 궤적 수(768)의 head-only GRPO를 dev64에서 평가 |

## 코드 구성

| 파일 | 역할 |
|---|---|
| `src/lsdrag/decision_policy.py` | `PolicyHead`, 에이전트 rollout, 행동 분포 |
| `src/lsdrag/decision_policy_recovery.py` | Qwen3-4B reader (prefill, 문법 제약 인자 생성) |
| `src/lsdrag/text_decision_benchmark.py` | 코퍼스 검색 도구(keyword/semantic/read_chunk) |
| `src/lsdrag/decision_grpo.py` | 그룹 상대 어드밴티지 |
| `src/lsdrag/lora4b.py` | LoRA 레이어와 샘플링 reader (변형 실험) |
| `scripts/repro_head_policy.py` | 재현용 spec 생성과 결과표 출력 |
| `scripts/run_decision_policy_pilot.py`, `scripts/fit_decision_state_heads.py` | 공용 경로·해시 함수와 head 로더. 다른 스크립트가 import하므로 직접 실행하지 않음 |

```bash
PYTHONPATH=src:scripts python -m pytest -q tests     # CPU 단위 테스트
```
