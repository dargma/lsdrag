# lsdrag: 도구 선택 head 학습 (BC → GRPO)

RAG 에이전트(Qwen3-4B-Instruct-2507)가 매 단계 쓸 도구를 LLM 대신 작은 head가 고르도록 학습합니다. 학습은 native 에이전트 모방(BC)으로 시작해 GRPO로 이어집니다. 이 문서 하나로 데이터 준비부터 평가까지 따라 할 수 있습니다.

## 결과 (MuSiQue 2hop dev64, greedy)

| 조건 | F1 (%) | 생성 토큰 | 입력 토큰 | 도구 호출 |
|---|---:|---:|---:|---:|
| baseline: LLM이 도구 선택 | 24.70 | 97.4 | 4,691 | 2.38 |
| BC head | 22.50 | 101.6 | 4,749 | 2.39 |
| GRPO head (KL 0.02, 1,536궤적) | 22.50 | 98.4 | 4,625 | 2.33 |
| GRPO head, KL 0 | TBD | TBD | TBD | TBD |
| GRPO head (KL 0.02, 768궤적) | TBD | TBD | TBD | TBD |
| GRPO head + LLM LoRA (KL 0.02, 768궤적) | TBD | TBD | TBD | TBD |

- **BC는 된다:** 학습 상태 667개 중 666개에서 native와 같은 도구를 고른다. 도구 호출 수도 baseline과 같다.
- **GRPO는 정확도를 바꾸지 못한다:** F1은 그대로이고 토큰만 약 3% 줄었다.
- **원인:** 비용 벌점이 정답 여부와 무관해서 쉬운 오답 경로만 짧아진다. 또 head는 다음 도구만 바꿀 뿐, 근거 읽기와 답 생성(LLM)은 건드리지 못한다.
- dev64에서 1문항은 F1 약 1.6%p다.

## 방법

**에이전트**
- 도구: `keyword_search`, `semantic_search`, `read_chunk`, `FINAL`
- 도구 호출 8회 뒤에는 FINAL을 강제한다.
- 매 단계 LLM이 컨텍스트를 읽는다. 도구는 head가 고르고, 인자와 답은 LLM이 문법 제약 아래 생성한다. LLM 가중치는 고정이다.

**head** (`src/lsdrag/decision_policy.py`의 `PolicyHead`, 약 34만 파라미터)
- 입력: LLM 마지막 층 마지막 토큰 hidden(2,560차원)
- 구조: Linear(2560→128) → Tanh → Linear(128→4)에 skip Linear(2560→4)를 더함
- 불가능한 행동은 마스크한다.

**BC**: native 행동의 음의 로그우도에 label smoothing 0.05를 더한다. 문항마다 가중치 합을 같게 맞춘다.

$$\mathcal{L}_{BC} = \sum_{s} w_s\Big[0.95 \cdot \big(-\log \pi_\theta(a^{t}_s \mid s)\big) + 0.05 \cdot \Big(-\frac{1}{|\mathcal{A}_s|}\sum_{a\in\mathcal{A}_s}\log\pi_\theta(a \mid s)\Big)\Big]$$

AdamW 3e-4, 전체 배치 100 step.

**GRPO**: 문항마다 4궤적을 샘플링하고, 같은 문항 안에서 표준화한 보상을 어드밴티지로 쓴다.

$$R_i = \mathrm{F1}_i - 0.03 \sum_{t} c_{i,t},\qquad c_{i,t} = \mathbf{1}[a_{i,t}\neq\mathrm{FINAL}] + \frac{g_{i,t}}{100} + \frac{p_{i,t}}{10000}$$

$$\mathcal{L} = \frac{1}{|B|}\sum_{(i,t)\in B} w_{i}\Big[-\min\big(\rho_{i,t} A_i,\ \mathrm{clip}(\rho_{i,t},0.8,1.2) \cdot A_i\big) + 0.02 \cdot \mathrm{KL}\big(\pi_\theta \parallel \pi_{BC}\big) - 0.01 \cdot H(\pi_\theta)\Big]$$

- c: 결정 하나의 비용(도구 호출 1, 생성 토큰 g, 입력 토큰 p)
- A: 같은 문항 4궤적 안에서 (R − 평균)/표준편차
- ρ: 수집 당시 대비 행동 확률 비율
- KL: BC head에서 멀어지는 정도에 대한 벌점
- H: 행동 엔트로피

학습은 action·skip 층만 한다. AdamW 1e-4, 192문항 × 4궤적 × 2회, 반복마다 4 epoch.

**LoRA 변형** (`scripts/run_lora_head_grpo.py`)
- head 업데이트는 위와 같다.
- 여기에 LLM의 attention·MLP 층에 LoRA(rank 16)를 붙인다.
- LLM이 temperature 1로 샘플링한 인자·답 토큰에 같은 어드밴티지로 PPO clip을 적용한다.

## 모델과 데이터

| 항목 | 출처 | 버전 |
|---|---|---|
| LLM | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | revision `cdbee75f17c01a7cc42f958dc650907174af0554` |
| 임베딩 | [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) | revision `914f7f89142e33e77833254d9c9b90c3cef7303b` |
| 데이터 | [MuSiQue](https://github.com/StonyBrookNLP/musique) answerable v1.0 train·dev | train SHA-256 `83a75b1e…`, dev `15fa6379…` |
| 채점 | MuSiQue 저장소의 `metrics/answer.py` (공식 F1) | |

**전처리 규칙**

| 대상 | 규칙 | 스크립트 |
|---|---|---|
| 코퍼스 | train·dev의 모든 paragraph를 모은다. (title, text)를 JSON으로 직렬화한 SHA-256으로 중복을 제거하고, `chunk_id = "c" + 해시 앞 24자`로 정한다. 정답과 support 정보는 쓰지 않는다. 결과는 101,962문단, `documents.jsonl` SHA-256 `60561e0e…` | `build_decision_shared_corpus.py` |
| 벡터 | 문서 입력은 `title: {title} \| text: {text}`, 질의 입력은 `task: search result \| query: {q}`. 768차원 L2 정규화 | `index_decision_shared_corpus.py` |
| 검색 | semantic은 벡터 내적, keyword는 단어 출현 점수. top_k 기본 5, 최대 20. 결과는 본문 앞 300자 | `src/lsdrag/text_decision_benchmark.py` |
| split | MuSiQue 2hop 문항 고정 목록: train 192 / validation(dev64) 64 / test 160 | `configs/split.json` |
| BC 데이터 | native 에이전트를 train192에서 greedy로 실행해, 강제 FINAL을 뺀 모든 결정의 (hidden, 가능 행동, 고른 행동)을 저장. 667개 | `collect_decision_shared_teacher.py` |

## 실행

경로는 고정입니다. 데이터와 실행 결과는 `/content/local_fast/lsdrag-text-v1`(`ROOT`), 임베딩 환경은 `/content/local_fast/lsdrag-v11`에 둡니다. GPU 1장이 필요합니다.

**1. 설치와 다운로드**

```bash
git clone -b jev_v2.0 https://github.com/dargma/lsdrag lsdrag-code && cd lsdrag-code
ROOT=/content/local_fast/lsdrag-text-v1; EMB=/content/local_fast/lsdrag-v11
mkdir -p $ROOT/hf $ROOT/source/musique $EMB/hf

# 시스템 torch(2.11, CUDA)를 그대로 쓰는 두 venv. 임베딩 쪽은 --no-deps 필수
uv venv --system-site-packages -p $(which python3) $ROOT/runtime
uv pip install -p $ROOT/runtime/bin/python "transformers==5.6.2" "lm-format-enforcer==0.11.3" jsonschema -e .
uv venv --system-site-packages -p $(which python3) $EMB/embedding-env
uv pip install --no-deps -p $EMB/embedding-env/bin/python "transformers==5.19.0" "sentence-transformers==6.1.0" -e .

hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --cache-dir $ROOT/hf
hf download google/embeddinggemma-2 --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --cache-dir $EMB/hf

git clone https://github.com/StonyBrookNLP/musique $ROOT/source/musique_scorer
gdown --id 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h --output $ROOT/musique_v1.0.zip
python -c "import zipfile;z=zipfile.ZipFile('$ROOT/musique_v1.0.zip');[open('$ROOT/source/musique/'+n.split('/')[-1],'wb').write(z.read(n)) for n in z.namelist() if n.endswith(('ans_v1.0_train.jsonl','ans_v1.0_dev.jsonl')) and 'MACOSX' not in n]"
```

**2. 전체 실행**

```bash
bash scripts/repro_pipeline.sh
```

아래 단계를 순서대로 실행하고, 끝난 단계는 건너뜁니다. 로그는 `$ROOT/logs/`, 결과는 `$ROOT/results/`에 생깁니다.

| 단계 | 스크립트 | 확인할 값 | 시간 |
|---|---|---|---|
| 코퍼스 | `build_decision_shared_corpus.py` | 101,962문단, SHA `60561e0e…` | CPU 수 분 |
| 벡터 | `index_decision_shared_corpus.py` | `[101962, 768]` | GPU 30분 안팎 |
| BC 데이터 | `collect_decision_shared_teacher.py` | 결정 상태 667개 근처, errors 0 | GPU 1시간 이내 |
| BC 학습 | `fit_decision_shared_initialization.py` | `train_accuracy` 0.9985 근처 | CPU 수 분 |
| GRPO | `run_decision_shared_grpo.py` | `curves.json`의 `changed_vs_anchor`가 작음 | GPU 약 1시간 |
| 평가 | `evaluate_decision_shared_corpus.py` | 마지막에 기준표와 재현표를 나란히 출력 | GPU 30분 이내 |

GPU 비결정성 때문에 숫자가 똑같이 나오지는 않습니다. 아래가 유지되면 재현된 것으로 봅니다.
- BC의 도구 호출 수가 baseline과 비슷하다.
- GRPO의 F1이 BC와 거의 같다.
- 두 head 모두 baseline보다 높지 않다.

**3. 변형 실험**

```bash
export PYTHONPATH=src:scripts; PY=$ROOT/runtime/bin/python
# head KL 계수 변경
$PY scripts/repro_head_policy.py grpo --name grpo_kl0 --kl 0
$PY scripts/run_decision_shared_grpo.py --manifest $ROOT/specs/grpo_kl0_manifest.json
$PY scripts/repro_head_policy.py eval --name eval_kl0 --grpo grpo_kl0 --heads-only
$PY scripts/evaluate_decision_shared_corpus.py --spec $ROOT/specs/eval_kl0_spec.json

# LLM LoRA + head (768궤적). 학습이 끝나면 LoRA+head와 같은 궤적 수의 head-only를 평가
$PY scripts/run_lora_head_grpo.py --name lora_grpo
bash scripts/lora_followup_eval.sh lora_grpo
```

## 파일

| 경로 | 내용 |
|---|---|
| `configs/experiment.json` | 모델·코퍼스 경로와 teacher·GRPO·평가 설정 |
| `configs/split.json` | train/validation/test 문항 ID |
| `configs/reference_dev64.json` | 위 결과표의 기준값 |
| `scripts/repro_head_policy.py` | 설정과 입력 파일 해시로 spec을 만들어 `$ROOT/specs`에 저장, 결과표 출력 |
| `scripts/common.py` | 공용 경로, 해시, head 로더 |
| `src/lsdrag/` | 에이전트, reader, 검색 도구, head, LoRA |
| `tests/` | `PYTHONPATH=src:scripts python -m pytest -q tests` |
