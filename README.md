# lsdrag: 도구 호출 효율화 GRPO (native ARAG vs select head ARAG)

RAG 에이전트(Qwen3-4B-Instruct-2507)를 GRPO로 학습함(보상: 정답일 때 도구 호출이 적을수록 높음). 실험 목적은 세 가지:
1. select head가 효율적 도구 호출의 정책이 될 수 있나
2. reader LLM과 head가 함께 있을 때 누가 얼마나 기여하나
3. LLM만 학습할 때와 LLM + head를 같이 학습할 때 둘 다 효과가 있다면, head가 돕는 이유는 무엇인가

이 문서만으로 데이터 준비부터 평가까지 진행 가능. 이전 head 정책 재현(BC → GRPO, 2hop dev64)은 브랜치 `jev_v2.0` README 참고.

## 질문과 답 (MuSiQue mixed-hop dev192 = 2/3/4hop 각 64문항, greedy)

판정은 사전에 고정한 iteration 4 기준. iteration 2는 학습 중간 결과로 함께 표시.

| 조건 | 학습 대상 | F1 (%) 학습 전 → it2 → it4 | ΔF1 it2 | ΔF1 it4 | 도구 호출 학습 전 → it2 → it4 | 출력 토큰 학습 전 → it2 → it4 | 빈 답 학습 전 → it2 → it4 |
|---|---|---|---|---|---|---|---|
| A1 native | LLM 전체 | 19.34 → 21.14 → 24.44 | +1.79 | +5.10 | 2.48 → 2.45 → 2.46 | 107.8 → 113.1 → 120.7 | 41% → 28% → 18% |
| B1 head + LLM | LLM 전체 + head 마지막 층 | 18.31 → 21.20 → 21.08 | +2.89 | +2.77 | 2.42 → 2.27 → 2.57 | 102.6 → 106.8 → 142.8 | 36% → 28% → 4% |
| C1 head만 | head 마지막 층 | 18.31 → 19.45 → 20.49 | +1.15 | +2.18 | 2.42 → 2.34 → 2.36 | 102.6 → 97.4 → 97.5 | 36% → 38% → 35% |
| C2 head만, lr 4배 | head 마지막 층 | 18.31 → 19.69 → 19.80 | +1.38 | +1.49 | 2.42 → 2.44 → 2.45 | 102.6 → 99.8 → 99.2 | 36% → 36% → 37% |

출력 토큰: 에피소드당 LLM 생성 토큰(검색 인자, read_chunk ID, 답, support_ids) 평균. 보상에 토큰 길이 항 없음. B1 답 길이 3.1 → 7.7단어(it4).

목표(사전 고정): 도구 호출 −10~30%, 정확도 동등 이상. iteration 2·4 모두 네 조건 미달.

**Q1. select head가 효율적 도구 호출 정책이 될 수 있나** → 이번 설정에서는 아님
- head만 학습(C1·C2): 정확도는 소폭 상승(+1.5~+2.2), 도구 호출은 변화 없음(−2.2 ~ +1.5%)
- lr 4배(C2)도 동일
- 용량 확인(아래 표 B): 고정 trunk 특징으로도 "근거를 다 보면 멈춤" 규칙 표현 가능. 마지막 층만 학습 AUC 0.867, trunk까지 학습 0.873. 병목은 표현 용량이 아니라 학습 신호

**Q2. LLM과 head가 함께 있을 때 누가 얼마나 기여하나** → 정확도는 LLM, 호출 효율은 둘 다 기여 없음
- 정확도: LLM만(A1) +5.10 > head만(C1) +2.18. LLM 학습의 주효과는 빈 답 감소(포기 대신 답함)
- 도구 호출(it4): A1 −0.8%, C1 −2.2%, B1 +6.5%. 어느 쪽도 목표만큼 줄이지 못함. it2의 B1(−6.0%, 유의)이 최대 감소

**Q3. LLM + head가 LLM만보다 나은가, head가 돕는 이유는** → 나은 근거 없음
- B1은 A1보다 정확도 향상이 작고(+2.77 vs +5.10) 도구 호출·생성 토큰이 늘어남(호출 +6.5%, 생성 토큰 102.6 → 142.8)
- native 결정 temperature 4.0 보정(결정 gradient 1/4)의 영향 확인(아래 표 A): native LLM의 결정은 head보다 더 많이 바뀜(argmax 변경 24% vs 15%). 결정 학습이 막힌 것은 아님. 다만 바뀐 방향이 P(FINAL) 감소(0.25 → 0.18)라서 호출이 줄지 않음

**교란 요인 확인** (`scripts/check_confounds.py`, 추가 학습 없이 저장된 모델과 rollout으로 측정)

A. 결정 정책 변화량 (dev192 학습 전 결정 상태 400개, 같은 상태에서 학습 전 → 후)

| 결정 주체 | argmax 변경 | 평균 \|Δp\| | P(FINAL) | 엔트로피 |
|---|---:|---:|---|---|
| native LLM (base → A1, T=1) | 24% | 0.216 | 0.248 → 0.181 | 0.157 → 0.202 |
| head (KL0 → C1) | 15% | 0.126 | 0.266 → 0.251 | 0.530 → 0.485 |

B. head 용량: 멈춤 규칙(근거 모두 확보 시 FINAL) 적합 (C1 train 결정 상태 4,000개, 문항 단위 80/20 분할, test AUC)

| head | 학습 파라미터 | test AUC | train AUC |
|---|---:|---:|---:|
| KL0 head, 학습 없음 | 0 | 0.849 | 0.740 |
| 마지막 층만 (action + skip) | 1.08만 | 0.867 | 1.000 |
| 전체 (trunk 포함) | 34만 | 0.873 | 1.000 |

상세(학습 추이, hop별, 토큰 구성, 정확도 변화 출처): `experiments/fullft_grpo_20261103/fullft_report.md`(it4), `fullft_report_it2.md`(it2)

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
- GRPO 학습 범위는 아래 표 참고

**학습 대상과 고정 부분**

| 조건 | LLM 4.02B (36층, 임베딩·lm_head 포함) | head trunk Linear(2560→128) 32.78만 | head action Linear(128→4) + skip Linear(2560→4) 1.08만 | head value·success·cost 0.12만 |
|---|---|---|---|---|
| A1 native | 학습 (전체) | 없음 | 없음 | 없음 |
| B1 head + LLM | 학습 (전체) | 고정 (BC 값) | 학습 | 미사용 |
| C1·C2 head만 | 고정 | 고정 (BC 값) | 학습 | 미사용 |

- LLM 학습: 모든 파라미터 업데이트. loss는 LLM이 샘플링한 토큰(native는 도구 이름 분기 토큰 포함)에만 걸리고, 문법이 강제한 토큰(`<tool_call>` 구문, `{"answer":` 등)은 loss에서 제외
- head trunk는 `jev_v2.0`의 BC 단계에서 학습된 값을 그대로 씀

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
| `RESULTS.md` | 실험 결과 문서(배경, 데이터셋, 학습 방법·수식, A1/B1/C1 결과, 논의) |
| `scripts/run_fullft_grpo.py` | GRPO 학습기. `--arm native\|head`, `--freeze-llm`, 보상 `discount\|latency\|calls\|process`, dev 평가, 체크포인트 |
| `src/lsdrag/fullft.py` | 샘플링 reader(두 arm), segment replay, fp32 master AdamW, 근거 단계 보상 |
| `scripts/make_mixhop_split.py`, `configs/split_mixhop.json` | mixed-hop split |
| `scripts/report_fullft.py` | 최종 보고서(표 1~8) |
| `scripts/diagnose_length_penalty.py` | 이전 GRPO에서 길이 penalty가 무효였던 원인 진단 |
| `scripts/check_head_execution.py`, `check_head_prefill_gpu.py`, `check_prefill_numerics.py` | head 실행 경로 검증(행동 실행, prefix prefill, KV 재사용 수치) |
| `scripts/calibrate_native_decision_temperature.py` | native 결정 T 보정(엔트로피 일치) |
| `scripts/compare_fullft_smoke.py` | smoke 비교표 |
| `scripts/validate_process_reward.py` | 단계 보상 오프라인 검증(실험 C3은 폐기) |
| `scripts/check_confounds.py` | 교란 요인 확인(결정 정책 변화량, head 용량) |
| `experiments/fullft_grpo_20261103/` | 설계 문서, 실행 스크립트, 중간·최종 보고서, 진단 결과 |
| `configs/`, `scripts/` 나머지, `tests/` | 이전 head 재현(`jev_v2.0` README), `PYTHONPATH=src:scripts python -m pytest -q tests` |
