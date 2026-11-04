# lsdrag: 도구 호출 효율화 GRPO (native ARAG vs select head ARAG)

RAG 에이전트(Qwen3-4B-Instruct-2507)를 GRPO로 학습해, 정확도는 유지하거나 높이면서 도구 호출(ARAG iteration)을 줄이는 실험. native(LLM이 도구 선택)와 select head(작은 head가 도구 선택) 두 방식을 비교함. 이 문서만으로 데이터 준비부터 평가까지 진행 가능. 이전 head 정책 재현(BC → GRPO, 2hop dev64)은 브랜치 `jev_v2.0`의 README 참고.

## 결과 (MuSiQue mixed-hop dev192 = 2/3/4hop 각 64문항, greedy)

| 조건 | 학습 | F1 (%) | EM (%) | 도구 호출 | 3회+ 비율 | 생성 토큰 | 입력 토큰 | 빈 답 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A0 native, 학습 전 | — | 19.34 | 14.58 | 2.48 | 52% | 107.8 | 5,029 | 41% |
| A1 native | LLM 전체 | 24.44 | 16.15 | 2.46 | 45% | 120.7 | 5,122 | 18% |
| B0/C0 head, 학습 전 | — | 18.31 | 13.02 | 2.42 | 42% | 102.6 | 4,902 | 36% |
| B1 head + LLM | LLM 전체 + head 마지막 층 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| C1 head만 | head 마지막 층 (LLM 고정) | 20.49 | 15.10 | 2.36 | 32% | 97.5 | 4,950 | 35% |
| C2 head만, lr 4배 | head 마지막 층 (LLM 고정) | 19.80 | 14.06 | 2.45 | 44% | 99.2 | 5,014 | 37% |

**학습 전 대비** (같은 arm의 iteration 0, 문항별 paired bootstrap 95% CI)

| 조건 | ΔF1 (%p) | Δ도구 호출 | 호출 변화율 | F1 오름/내림/같음 |
|---|---|---|---:|---|
| A1 native | +5.10 [+0.66, +9.72] | −0.02 [−0.11, +0.08] | −0.8% | 27 / 12 / 153 |
| B1 head + LLM | TBD | TBD | TBD | TBD |
| C1 head만 | +2.18 [+0.08, +4.63] | −0.05 [−0.13, +0.04] | −2.2% | 7 / 4 / 181 |
| C2 head만, lr 4배 | +1.49 [+0.07, +3.32] | +0.04 [−0.01, +0.08] | +1.5% | 4 / 2 / 186 |

판정 기준(사전 고정): iteration 4에서 도구 호출 −10~30%, ΔF1 ≥ 0 이고 CI 하한 > −3%p.

## 방법

**에이전트**: native A-RAG([Ayanami0730/arag](https://github.com/Ayanami0730/arag), revision `a44de6b`) 참고
- 도구: `keyword_search`, `semantic_search`, `read_chunk`, `FINAL`. 도구 호출 8회 이후 FINAL 강제
- 매 단계 전체 컨텍스트를 다시 prefill. 인자와 답은 LLM이 문법 제약 하에 생성
- native: LLM이 도구 이름 토큰을 선택(합법 도구 trie 분기)
- head: `PolicyHead`가 LLM 마지막 층 hidden으로 도구를 고르고, 해당 tool-call 구문을 prefill한 뒤 LLM이 인자만 생성
- 두 arm 모두 FINAL은 `{"answer":`를 prefill

**head** (`src/lsdrag/decision_policy.py`의 `PolicyHead`, 33.97만 파라미터, LLM 대비 0.008%)
- trunk Linear(2560→128)+Tanh → action Linear(128→4), skip Linear(2560→4)를 더함
- 시작점: `jev_v2.0`의 GRPO head, KL 0 (2hop 학습)
- GRPO에서는 action·skip 층(1.08만)만 학습, trunk 고정

**보상과 어드밴티지**

$$R_i = \mathrm{F1}_i \cdot 0.8^{\,n_i},\qquad A_i = R_i - \overline{R}_{\text{group}}$$

- n: 도구 호출 수. 맞혔을 때만 호출이 적을수록 이득, 오답은 0
- 같은 문항 8궤적 평균을 뺌(std 나눗셈 없음)

**loss** (두 arm 공통, 반복마다 A≠0 궤적 1 pass, 미니배치 16궤적)

$$\mathcal{L} = \frac{1}{16}\sum_{i}\Big[\sum_{t\in\text{결정}} \ell(\rho_{i,t},A_i) + \sum_{k\in\text{free 토큰}} \ell(\rho_{i,k},A_i) + 0.02\cdot \mathrm{KL}_{k3}(\pi_{\text{ref}}\,\|\,\pi_\theta)\Big],\quad \ell(\rho,A) = -\min\big(\rho A,\ \mathrm{clip}(\rho,0.8,1.2) \cdot A\big)$$

- 결정: native는 도구 이름 분기 토큰 logprob 합, head는 head logprob
- free 토큰: LLM이 샘플링한 인자·답 토큰. 문법이 강제한 토큰은 제외
- ref: 각 arm의 iteration 0 정책. 엔트로피 bonus 없음

**학습 설정**
- rollout: 반복마다 train pool에서 hop별 64문항씩 192문항 추출 × 8궤적, 4회 반복
- 샘플링: 인자·답 T=1. 결정 T는 native 4.0(head 엔트로피 0.46에 맞춤), head 1.0
- LLM: 4.02B 전체, fp32 master 가중치 + fp32 AdamW, lr 1e-6, gradient checkpointing (GPU 약 70GB)
- head lr 5e-5 (C2는 2e-4)
- 평가: dev192 greedy, iteration 0·2·4. 판정은 iteration 4

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
| 코퍼스 | train·dev 전체 paragraph, (title, text) JSON의 SHA-256으로 중복 제거. 101,962문단, `documents.jsonl` SHA-256 `60561e0e…` | `build_decision_shared_corpus.py` |
| 벡터 | 문서 `title: {title} \| text: {text}`, 질의 `task: search result \| query: {q}`. 768차원 L2 정규화 | `index_decision_shared_corpus.py` |
| 검색 | semantic은 벡터 내적, keyword는 단어 출현 점수. top_k 기본 5, 최대 20. 본문 앞 300자 | `src/lsdrag/text_decision_benchmark.py` |
| split | 2hop은 `configs/split.json`과 동일. 3·4hop은 MuSiQue answerable train에서 하위 유형별 층화 추출(seed 20261103). train 576 / validation 192 / test 320(보류), hop별 균등 | `make_mixhop_split.py` → `configs/split_mixhop.json` |

## 실행

경로 고정. 데이터와 결과는 `/content/local_fast/lsdrag-text-v1`(`ROOT`), 임베딩 환경은 `/content/local_fast/lsdrag-v11`. GPU 1장(A1·B1 약 70GB, C 약 9GB).

**1. 설치, 다운로드, 코퍼스·head 준비**: `jev_v2.0` README의 1~2단계 그대로 실행(`bash scripts/repro_pipeline.sh`). 이어서 KL 0 head 생성

```bash
export PYTHONPATH=src:scripts; PY=$ROOT/runtime/bin/python
$PY scripts/repro_head_policy.py grpo --name grpo_kl0 --kl 0
$PY scripts/run_decision_shared_grpo.py --manifest $ROOT/specs/grpo_kl0_manifest.json   # → results/decision_shared_grpo192_kl0/grpo_iteration1.pt
```

**2. split 생성**

```bash
python scripts/make_mixhop_split.py   # configs/split_mixhop.json (저장소에 포함, 재생성 시 동일)
```

**3. 학습과 평가** (공통 인자)

```bash
C="--split configs/split_mixhop.json --resample --questions 192 --iterations 4 --repeats 8 --reward discount --gamma 0.8 --beta 0.02 --eval-at 0,2,4"
$PY scripts/run_fullft_grpo.py --arm native --name mh_native_A1 --decision-temperature 4.0 --lr 1e-6 --save-model-at 4 $C
$PY scripts/run_fullft_grpo.py --arm head --name mh_head_B1 --lr 1e-6 --lr-head 5e-5 --save-model-at 4 $C
$PY scripts/run_fullft_grpo.py --arm head --freeze-llm --name mh_headonly_C1 --lr-head 5e-5 $C
$PY scripts/run_fullft_grpo.py --arm head --freeze-llm --name mh_headonly_C2_lr2e-4 --lr-head 2e-4 $C
```

| 단계 | 시간 (arm당) | 확인 값 |
|---|---|---|
| iteration 0 평가 | 약 12분 | native F1 19.34 / 호출 2.48, head F1 18.31 / 호출 2.42 |
| iteration 1회 (192×8 rollout + 업데이트) | 약 1.5시간 | `curves.json`의 `kl` 1 이하(일시적 급등 가능), `gap` 약 0.02 |
| 전체 4회 + 평가 2회 | A1·B1 약 6.5시간, C 약 5시간 | `eval/i4/summary.json` |

**4. 보고서**

```bash
python scripts/report_fullft.py --out $ROOT/analysis/fullft_report.md   # 표 1~8: 주 결과, paired 판정, 학습 추이, 호출 분포, 토큰 구성, 정확도 변화 출처, 건전성, hop별 효율
```

## 파일

| 경로 | 내용 |
|---|---|
| `scripts/run_fullft_grpo.py` | GRPO 학습기. `--arm native\|head`, `--freeze-llm`, 보상 `discount\|latency\|calls\|process`, dev 평가, 체크포인트 |
| `src/lsdrag/fullft.py` | 샘플링 reader(두 arm), segment replay, fp32 master AdamW, 근거 단계 보상 |
| `scripts/make_mixhop_split.py`, `configs/split_mixhop.json` | mixed-hop split |
| `scripts/report_fullft.py` | 최종 보고서(표 1~8) |
| `scripts/diagnose_length_penalty.py` | 이전 GRPO에서 길이 penalty가 무효였던 원인 진단 |
| `scripts/check_head_execution.py`, `check_head_prefill_gpu.py`, `check_prefill_numerics.py` | head 실행 경로 검증(행동 실행, prefix prefill, KV 재사용 수치) |
| `scripts/calibrate_native_decision_temperature.py` | native 결정 T 보정(엔트로피 일치) |
| `scripts/compare_fullft_smoke.py` | smoke 비교표 |
| `scripts/validate_process_reward.py` | 단계 보상 오프라인 검증(실험 C3은 폐기) |
| `experiments/fullft_grpo_20261103/` | 설계 문서, 실행 스크립트, 중간·최종 보고서, 진단 결과 |
| `configs/`, `scripts/` 나머지, `tests/` | 이전 head 재현(`jev_v2.0` README), `PYTHONPATH=src:scripts python -m pytest -q tests` |
