# lsdrag: 도구 선택 head 학습 (BC → GRPO)

RAG 에이전트(Qwen3-4B-Instruct-2507)에서 매 단계 쓸 도구를 LLM 대신 작은 head가 선택하도록 학습함. native 에이전트 모방(BC) 후 GRPO 적용. 이 문서만으로 데이터 준비부터 평가까지 진행 가능.

## 결과 (MuSiQue 2hop dev64, greedy)

| 조건 | F1 (%) | 생성 토큰 | 입력 토큰 | 도구 호출 |
|---|---:|---:|---:|---:|
| baseline: LLM이 도구 선택 | 24.70 | 97.4 | 4,691 | 2.38 |
| BC head | 22.50 | 101.6 | 4,749 | 2.39 |
| GRPO head (KL 0.02, 1,536궤적) | 22.50 | 98.4 | 4,625 | 2.33 |
| GRPO head, KL 0 (1,536궤적) | 25.97 | 99.1 | 4,585 | 2.31 |
| GRPO head + LLM LoRA, KL 0 (1,536궤적) | TBD | TBD | TBD | TBD |

## 방법

**에이전트**: native A-RAG([Ayanami0730/arag](https://github.com/Ayanami0730/arag), revision `a44de6b`) 참고
- 도구: `keyword_search`, `semantic_search`, `read_chunk`, `FINAL`
- 도구 호출 8회 이후 FINAL 강제
- 매 단계 LLM이 컨텍스트를 읽음. 도구 선택은 head, 인자와 답 생성은 LLM(문법 제약). LLM 가중치는 고정

**head** (`src/lsdrag/decision_policy.py`의 `PolicyHead`, 약 34만 파라미터)
- 입력: LLM 마지막 층 마지막 토큰 hidden(2,560차원)
- 구조: Linear(2560→128) → Tanh → Linear(128→4), 여기에 skip Linear(2560→4)를 더함
- 불가능한 행동은 마스크

**BC**: native 행동의 음의 로그우도 + label smoothing 0.05. 문항별 가중치 합을 동일하게 맞춤

$$\mathcal{L}_{BC} = \sum_{s} w_s\Big[0.95 \cdot \big(-\log \pi_\theta(a^{t}_s \mid s)\big) + 0.05 \cdot \Big(-\frac{1}{|\mathcal{A}_s|}\sum_{a\in\mathcal{A}_s}\log\pi_\theta(a \mid s)\Big)\Big]$$

AdamW 3e-4, 전체 배치 100 step.

**GRPO**: 문항마다 4궤적 샘플링. 같은 문항 안에서 표준화한 보상을 어드밴티지로 사용

$$R_i = \mathrm{F1}_i - 0.03 \sum_{t} c_{i,t},\qquad c_{i,t} = \mathbf{1}[a_{i,t}\neq\mathrm{FINAL}] + \frac{g_{i,t}}{100} + \frac{p_{i,t}}{10000}$$

$$\mathcal{L} = \frac{1}{|B|}\sum_{(i,t)\in B} w_{i}\Big[-\min\big(\rho_{i,t} A_i,\ \mathrm{clip}(\rho_{i,t},0.8,1.2) \cdot A_i\big) + 0.02 \cdot \mathrm{KL}\big(\pi_\theta \parallel \pi_{BC}\big) - 0.01 \cdot H(\pi_\theta)\Big]$$

- c: 결정 하나의 비용(도구 호출 1, 생성 토큰 g, 입력 토큰 p)
- A: 같은 문항 4궤적 안에서 (R − 평균)/표준편차
- ρ: 수집 당시 대비 행동 확률 비율
- KL: BC head에서 멀어진 정도에 대한 벌점
- H: 행동 엔트로피

action·skip 층만 학습. AdamW 1e-4, 192문항 × 4궤적 × 2회, 반복마다 4 epoch.

**LoRA 변형** (`scripts/run_lora_head_grpo.py`)
- head 업데이트는 위 GRPO와 동일(KL 계수는 `--kl`로 지정, 결과표 실행은 0)
- LLM의 attention·MLP 층에 LoRA(rank 16) 추가
- LLM이 temperature 1로 샘플링한 인자·답 토큰에 같은 어드밴티지로 PPO clip 적용
- 궤적 수, 보상, seed는 GRPO와 동일(192문항 × 4궤적 × 2회)
- LoRA는 반복마다 1 pass, AdamW 2e-5. base LLM에 대한 KL 없음

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
| 코퍼스 | train·dev의 모든 paragraph 수집. (title, text)를 JSON 직렬화한 SHA-256으로 중복 제거, `chunk_id = "c" + 해시 앞 24자`. 정답·support 정보는 미사용. 101,962문단, `documents.jsonl` SHA-256 `60561e0e…` | `build_decision_shared_corpus.py` |
| 벡터 | 문서 입력 `title: {title} \| text: {text}`, 질의 입력 `task: search result \| query: {q}`. 768차원 L2 정규화 | `index_decision_shared_corpus.py` |
| 검색 | semantic은 벡터 내적, keyword는 단어 출현 점수. top_k 기본 5, 최대 20. 결과는 본문 앞 300자 | `src/lsdrag/text_decision_benchmark.py` |
| split | MuSiQue 2hop 문항 고정 목록: train 192 / validation(dev64) 64 / test 160 | `configs/split.json` |
| BC 데이터 | native 에이전트를 train192에서 greedy 실행. 강제 FINAL을 제외한 모든 결정의 (hidden, 가능 행동, 선택 행동) 저장. 667개 | `collect_decision_shared_teacher.py` |

## 실행

경로 고정. 데이터와 실행 결과는 `/content/local_fast/lsdrag-text-v1`(`ROOT`), 임베딩 환경은 `/content/local_fast/lsdrag-v11`. GPU 1장 필요.

**1. 설치와 다운로드**

```bash
git clone -b jev_v2.0 https://github.com/dargma/lsdrag lsdrag-code && cd lsdrag-code
ROOT=/content/local_fast/lsdrag-text-v1; EMB=/content/local_fast/lsdrag-v11
mkdir -p $ROOT/hf $ROOT/source/musique $EMB/hf

# 시스템 torch(2.11, CUDA)를 그대로 쓰는 venv 2개. 임베딩 쪽은 --no-deps 필수
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

아래 단계를 순서대로 실행하며, 완료된 단계는 건너뜀. 로그는 `$ROOT/logs/`, 결과는 `$ROOT/results/`에 저장.

| 단계 | 스크립트 | 확인 값 | 시간 |
|---|---|---|---|
| 코퍼스 | `build_decision_shared_corpus.py` | 101,962문단, SHA `60561e0e…` | CPU 수 분 |
| 벡터 | `index_decision_shared_corpus.py` | `[101962, 768]` | GPU 30분 안팎 |
| BC 데이터 | `collect_decision_shared_teacher.py` | 결정 상태 약 667개, errors 0 | GPU 1시간 이내 |
| BC 학습 | `fit_decision_shared_initialization.py` | `train_accuracy` 약 0.9985 | CPU 수 분 |
| GRPO | `run_decision_shared_grpo.py` | `curves.json`의 `changed_vs_anchor`가 작음 | GPU 약 1시간 |
| 평가 | `evaluate_decision_shared_corpus.py` | 마지막에 기준표와 재현표를 나란히 출력 | GPU 30분 이내 |

GPU 비결정성으로 수치는 완전히 같지 않음. dev64에서 1문항은 F1 약 1.6%p. 아래 세 가지가 유지되면 재현으로 판정.
- BC의 도구 호출 수가 baseline과 비슷함
- GRPO의 F1이 BC와 거의 같음
- 두 head 모두 baseline보다 높지 않음

**3. 변형 실험**

```bash
export PYTHONPATH=src:scripts; PY=$ROOT/runtime/bin/python
# head KL 계수 변경
$PY scripts/repro_head_policy.py grpo --name grpo_kl0 --kl 0
$PY scripts/run_decision_shared_grpo.py --manifest $ROOT/specs/grpo_kl0_manifest.json
$PY scripts/repro_head_policy.py eval --name eval_kl0 --grpo grpo_kl0 --heads-only
$PY scripts/evaluate_decision_shared_corpus.py --spec $ROOT/specs/eval_kl0_spec.json

# LLM LoRA + head (1,536궤적). 학습 완료 후 dev64 평가
$PY scripts/run_lora_head_grpo.py --name lora_grpo --kl 0
bash scripts/lora_followup_eval.sh lora_grpo
```

## 파일

| 경로 | 내용 |
|---|---|
| `configs/experiment.json` | 모델·코퍼스 경로, teacher·GRPO·평가 설정 |
| `configs/split.json` | train/validation/test 문항 ID |
| `configs/reference_dev64.json` | 결과표 기준값 |
| `scripts/repro_head_policy.py` | 설정과 입력 파일 해시로 spec 생성(`$ROOT/specs`), 결과표 출력 |
| `scripts/common.py` | 공용 경로, 해시, head 로더 |
| `src/lsdrag/` | 에이전트, reader, 검색 도구, head, LoRA |
| `tests/` | `PYTHONPATH=src:scripts python -m pytest -q tests` |
