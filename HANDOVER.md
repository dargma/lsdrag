# 인계서: lsdrag full-FT GRPO 실험 (2026-11-04 09:4x UTC 기준)

## 0. 한눈에

| 항목 | 상태 |
|---|---|
| 주 실험 (A1 native, B1 head + LLM, C1 head만, C2 head만·lr 4배) | **완료** (dev192, iteration 4 판정) |
| 결과 문서 | `RESULTS.md`(배경·데이터·수식·결과·논의), `README.md`(재현 가이드) |
| 교란 요인 확인 (native 결정 T, head 용량) | **완료** (`scripts/check_confounds.py`) |
| 토큰 제약 재실험 (B2, A2) | **중단**. B2는 iteration 1 수집 160/192에서 사용자 지시로 정지. A2는 미시작 |
| 단계별 보상 C3 | 사용자 지시로 폐기(코드와 오프라인 검증만 남음) |
| head trunk까지 학습 (C1-full, B1-full) | 사용자 결정으로 실행 안 함("그냥 지금대로") |
| 실행 중 프로세스 | 없음. 남은 GPU 프로세스는 다른 프로젝트(phonebot) 것으로 건드리지 않음 |
| 결론 | 네 arm 모두 정확도는 상승. 도구 호출 −10~30% 목표는 미달 |

## 1. 위치

| 항목 | 위치 |
|---|---|
| 코드 (이번 실험) | GitHub `dargma/lsdrag` 브랜치 **`jev_v2.1`** (최신 커밋은 `git log -1`) |
| 코드 (이전 head 재현, 정리본) | 브랜치 `jev_v2.0` (README만 보고 재현 가능하게 정리된 상태, 수정 금지 권장) |
| 로컬 작업 경로 | 코드 `/content/local_fast/lsdrag-code`. 데이터·결과 `ROOT=/content/local_fast/lsdrag-text-v1`. 임베딩 env `/content/local_fast/lsdrag-v11` |
| 결과 디렉터리 | `ROOT/results/{mh_native_A1, mh_head_B1, mh_headonly_C1, mh_headonly_C2_lr2e-4}` |
| 중단·폐기 run | `ROOT/results/_stopped*`, `_aborted*`, `smoke_*` |
| 분석 산출물 | `ROOT/analysis/` (설계 문서 `grpo_length_design.md`가 전체 의사결정 기록) |
| 실험 기록 사본 | repo `experiments/fullft_grpo_20261103/` (설계 문서, 보고서, 실행 스크립트, 진단 JSON) |
| Drive 백업 | `MyDrive/lsdrag-backups/20261104/` (아래 3절). 이전 원본 `20261023/`, `20261028/` |
| HF | `sungkwang2/lsdrag-v2-head-policy` (private, 이전 BC·GRPO head 가중치). 데이터 repo는 quota 문제로 비어 있음(보류) |

## 2. 결과 요약 (dev192 = 2/3/4hop 각 64문항, greedy)

| 조건 | 학습 위치 | F1 학습 전 → it2 → it4 | 도구 호출 학습 전 → it2 → it4 | 출력 토큰 학습 전 → it4 | 빈 답 학습 전 → it4 |
|---|---|---|---|---|---|
| A1 native | LLM 전체 | 19.34 → 21.14 → 24.44 | 2.48 → 2.45 → 2.46 | 107.8 → 120.7 | 41% → 18% |
| B1 head + LLM | LLM 전체 + head 출력층 | 18.31 → 21.20 → 21.08 | 2.42 → 2.27 → 2.57 | 102.6 → 142.8 | 36% → 4% |
| C1 head만 | head 출력층 | 18.31 → 19.45 → 20.49 | 2.42 → 2.34 → 2.36 | 102.6 → 97.5 | 36% → 35% |
| C2 head만, lr 4배 | head 출력층 | 18.31 → 19.69 → 19.80 | 2.42 → 2.44 → 2.45 | 102.6 → 99.2 | 36% → 37% |

**핵심 관찰**
- 정확도 기여는 LLM 학습이 큼. 주효과는 빈 답(포기) 감소
- B1은 it2에서 호출 −6.0%(유의)였으나 it4에서 +6.5%로 반전함. 출력 토큰 +39%, 답 2.5배 길어짐
  - 원인: F1 부분 점수를 노린 장황한 답. 예: `3hop1__566343_854928_670567`
- 교란 요인 확인
  - native 결정도 학습됨: argmax 변경 24% vs head 15%
  - head 용량은 병목 아님: 멈춤 규칙 AUC 0.867(출력층만) vs 0.873(trunk 포함)
- 병목은 학습 신호
  - 궤적 단위 보상이라 결정의 51%가 신호 0을 받음
  - F1 부분 점수 때문에 길게라도 답하는 것이 유리함
  - 토큰 길이 항이 없음

**중단된 B2 (토큰 제약)**
- 학습 전 dev192: F1 18.11, 호출 2.42, 출력 토큰 101.7. B1 학습 전과 같고, 60자 상한 영향은 F1 −0.2
- iteration 1 rollout 1,348궤적(업데이트 전): F1 18.4, 호출 2.50, 출력 98.2
- 학습 업데이트는 한 번도 안 됨 → 결과로 쓸 수 없음

## 3. Drive 백업 목록 (`MyDrive/lsdrag-backups/20261104/`, SHA256SUMS_all 포함)

| 파일 | 내용 |
|---|---|
| `mh_native_A1.tar.gz`, `mh_head_B1.tar.gz`, `mh_headonly_C1.tar.gz`, `mh_headonly_C2_lr2e-4.tar.gz` | run 전체(episodes, eval i0/i2/i4, curves, head 체크포인트). LLM 가중치는 제외 |
| `mh_native_A1_model_iteration4.tar`, `mh_head_B1_model_iteration4.tar` | iteration 4 LLM 가중치(각 8GB, safetensors) |
| `_stopped_mh_head_B2_tok.tar.gz`, `_stopped2_mh_head_B2_tok.tar.gz` | 중단된 B2 두 번(1차는 다른 프로젝트 GPU 점유로, 2차는 사용자 정지로 중단) |
| `_aborted_2hop_*`, `smoke_*` | 2hop에서 중단된 run, smoke 테스트 |
| `analysis.tar.gz` | `ROOT/analysis`, 실행 스크립트(`run_mixhop_*.sh`), watcher, `logs/` |
| `confound_checks.json` | 교란 요인 확인 결과 |

iteration 2 LLM 가중치는 저장하지 않음(`--save-model-at 4`만 지정). it2 모델은 다시 만들 수 없음.

## 4. 새 런타임에서 복원

1. 코드: `git clone -b jev_v2.1 https://github.com/dargma/lsdrag /content/local_fast/lsdrag-code`
2. 환경과 모델·데이터: `jev_v2.0` README 1단계 그대로 진행
   - uv venv 2개. embedding env는 `--no-deps`
   - Qwen3-4B·embeddinggemma는 고정 revision
   - MuSiQue zip과 scorer clone
3. 코퍼스·벡터·BC·KL0 head
   - 공개 데이터로 다시 만들 수 있음: `bash scripts/repro_pipeline.sh` 후 KL0 GRPO
   - 시간을 아끼려면 Drive 원본에서 복원
     - `20261023/`: `musique_shared_20261023_saved`, `decision_shared_teacher192_20261023`, `decision_shared_initialization_20261023`, `decision_shared_grpo192_20261023`, `decision_shared_validation64_20261023`
     - `20261028/`: `decision_shared_grpo192_kl0`, `decision_shared_validation64_kl`
   - 원본 SHA 앞자리: 문서 `60561e0e`, 벡터 `e2b7013c`, teacher `a73873f7`, BC head `0023dd45`
4. 이번 실험 결과가 필요하면 `20261104/`의 tar를 `ROOT/results`에 풂
5. 점검: `bash ROOT/verify_restore.sh`(해시·테스트), `PYTHONPATH=src:scripts python -m pytest -q tests`(18 pass)

## 5. 코드 맵 (이번 실험)

| 파일 | 역할 |
|---|---|
| `scripts/run_fullft_grpo.py` | 학습기. `--arm native\|head`, `--freeze-llm`, `--reward discount\|latency\|calls\|process`, `--gamma`, `--token-gamma`, `--answer-max-chars`, `--decision-temperature`, `--split`, `--resample`, `--eval-at`, `--save-model-at` |
| `src/lsdrag/fullft.py` | `SampledReader`(native 분기 샘플링, payload 샘플링, FINAL prefix, 답 길이 상한), `segment_logps`(replay), `MasterAdamW`(fp32 master), `evidence_step_rewards`(단계 보상) |
| `scripts/make_mixhop_split.py` → `configs/split_mixhop.json` | mixed-hop split(seed 20261103) |
| `scripts/report_fullft.py` | 표 1~8 보고서(`--final 2\|4`) |
| `scripts/check_confounds.py` | 교란 요인 확인 |
| `scripts/validate_process_reward.py` | 단계 보상 오프라인 검증 |
| `scripts/diagnose_length_penalty.py`, `check_head_*`, `check_prefill_numerics.py`, `calibrate_native_decision_temperature.py`, `compare_fullft_smoke.py` | 진단 |
| `ROOT/watch_fullft.py` | run 이벤트 감시(Monitor용). `RUNS` 목록을 수정해 사용 |
| `experiments/fullft_grpo_20261103/run_mixhop_*.sh` | 실제 실행에 쓴 launch 스크립트 |

## 6. 남은 일 (우선순위)

1. **토큰 제약 재실험 B2 → A2**: `bash ROOT/run_mixhop_tok.sh`(각 약 3.5시간, 순차)
   - 보상 F1·0.8^호출 수·0.995^생성 토큰, 답 60자 상한, 2 iteration
   - 비교 대상: B1·A1의 it2
   - 실행 전 결과 디렉터리 이름 충돌 확인(`mh_head_B2_tok`은 지금 없음)
   - GPU가 비었는지 확인할 것. 업데이트 때 약 70GB 필요
2. **업데이트 속도 개선**: 같은 궤적의 segment를 1회 forward로 합치기, 길이별 배치. 현재 iteration당 업데이트 약 20분
3. **iteration마다 LLM 가중치 저장**과 test320 검증. it2 대 it4 정지 시점 문제를 해결하기 위함
4. **단계별 보상**(`--reward process`, head-only에서만 구현): 오프라인 검증에서 신호 0인 결정이 51% → 11%로 줄어듦
5. **보상에서 F1 대신 EM 사용**, 또는 답 길이 상한 유지

## 7. 주의사항 (겪은 문제)

- **Drive hook**
  - Drive는 쓰기 전용 백업으로 취급함. 셸 cwd가 Drive 경로일 때 `tar`·`sha256sum`·`du`·`mv` 등이 hook에 차단됨
  - 해결: `cd /content/local_fast/...`에서 실행하거나, 스크립트 파일로 만들어 실행
- **다른 프로젝트 GPU 작업**(CARLA, pm_native_env/cem_v1, phonebot): 절대 종료하지 않음. 공유 중에는 rollout이 최대 2배 느려지고, 업데이트 시 OOM 위험이 있음
- **`pkill -f`**: 패턴이 자기 셸 명령과 겹치면 셸까지 죽음(exit 144). PID로 kill 할 것
- **bf16 stochastic rounding optimizer**(torchao): 잡음 때문에 KL이 폭주해서 쓰지 않음. fp32 master `MasterAdamW` 사용(메모리 약 70GB)
- **KL 일시 급등**: A1 it1에서 5.5e6, it3에서 4.6이 나왔지만 다음 iteration에 정상으로 돌아옴. replay gap은 계속 약 0.02로 정상
- **실행 중 스크립트 수정 금지**: bash는 스크립트를 순차로 읽음. 실행 중인 `.sh`는 임시 파일에 쓰고 `mv`로 교체할 것. python은 import 이후 수정해도 안전함
- **head-only arm(C*)의 iteration 0 평가**: C1과 동일하므로 report가 C1 i0을 대신 씀(`report_fullft.py`의 fallback)
- **wall-clock 시간**: GPU를 공유한 run끼리는 비교 불가. 시간은 지연 proxy(0.026·생성 + 0.218·입력 k토큰, R² 0.98)로 비교
- **head는 trunk 고정**: GRPO는 action·skip(1.08만 파라미터)만 학습함

## 8. 사용자 선호

- 보고는 한글로, 표 위주로 간결하게. 문서 문체는 "~함/~음"
- 결과 표에 신뢰구간 범위·변화율 괄호를 넣지 않음. 표 아래에 해석 문장을 넣지 않음(README)
- 수식은 곱하기에 `\cdot`을 씀. `\,` 같은 띄어쓰기 명령은 쓰지 않음(콤마처럼 렌더링됨)
- `jev_v2.0`은 README만으로 재현할 수 있게 정리된 브랜치. 새 작업은 `jev_v2.1`에 push
- git 커밋 작성자는 `sungkwang2 <sungkwang2@gmail.com>`
- 실험 판정 시점은 사전에 고정함. 결과를 본 뒤 시점을 바꾸면 사후 관찰로 표시
- 새 실험은 오프라인 검증 후 "진짜 필요하고 효과가 예상되면" 실행
