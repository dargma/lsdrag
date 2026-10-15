# Active phase — first2 diagnostic launched

PID16960 launched fresh native/generated/first-token paired diagnostic, first2 fixed train questions. Launch record `/content/local_fast/lsdrag-head-v13/launch.json`; log `/content/local_fast/lsdrag-head-v13/logs/diagnostic_000_002.log`. Verify live process/command before assuming it remains active. Preflight all-five-tools/FINAL/images passed; no quality claim. Source frozen at604731a. Next inspect6 expected episode artifacts and infra/format behavior, then extend unchanged first20. Diagnostic backup bundles every completed question.

# 현재 실행 — 기존 인터페이스 진단부터 순차 진행

사용자가 검토안 순차 실행과 단계별 저장을 승인했다. 현 계약: `docs/JEV_DECISION_RAG_EXECUTION_20261015.md`.1단계 train20 native/generated/first-token paired 진단,2단계 fresh60/80/100 coverage-gated3-seed learned head+dev20 평가. 인자 최소화는 독립 후속 ablation. 이전 argument-candidate 결합안은 실행하지 않는다.

Run `/content/local_fast/lsdrag-head-v13/results/decision_rag_20261015`, common args `lab/registry/decision_rag_20261015/common_args.json`. Native parser dependency를 vLLM0.20.0의 unchanged typed helpers로 분리하여19 CPU tests 통과. 현재 첫 실모델 smoke/first2 준비 중. 새 실험 아직 완료 아님.

# 최신 검토 — 품질/전체 효율을 기준으로 설계 재평가

사용자는 직전 제안에 쏠리지 말고 RAG+decision 결합의 성능 향상/효율화를 독립적으로 검토하도록 요청했다. `docs/JEV_DECISION_RAG_REASSESSMENT_20261015.md`에 문헌·코드·과거20문항 report를 대조한 검토를 저장했다. 권고는 A-RAG 전면 재설계/추가 인자제어기를 바로 붙이지 않고, 기존 interface에서 decision의 추가 가치를 검증한 뒤 인자 최소화를 별도 ablation으로 보는 것. 이는 새 실행 계획 동결이 아니라 연구 검토다. 기존 전체 목표(teacher/학습/dev 포함)는 유지하며 새 코드/실험은 아직 시작하지 않았다. 과거 보고서 수치를 새 결과로 주장하거나 라벨로 재사용하지 않는다.

# 최신 steering — 설계 타당성 재검토

사용자는 전체 범위 승인 직후, 인자 최소화가 구성요소를 덕지덕지 붙이는 방향인지, 작은 head가 판단 가능한지, tool call 최소화 목적에 맞는지, A-RAG 인터페이스 재설계가 필요한지 질문했다. 범위 축소나 목표 취소가 아니라 현재 설계에 대한 재검토 요청이다. 확정 범위만851925a로 기록했으며 인자 제어 코드/새 실험은 아직 시작하지 않았다. 먼저 목표(호출 수 vs 인자 생성 vs 전체 계산)와 학습 target(teacher imitation vs 비용 대비 QA 가치)을 분리해 검토한다. 기존 다섯 도구 의미를 보존하면서 가능한 typed execution option을 공동 선택하는 인터페이스는 검토안이며 아직 승인된 대체 구현이 아니다.

# 최신 승인 — 2026-10-15

사용자가 “모두 포함”으로 새 설계와2단계의 teacher 수집·head 학습·dev 평가 포함 범위를 확정했다. 이전 승인 대기 문구는 해제한다. `docs/JEV_ARGUMENT_CONTROL_PLAN_20261015.md`의 확정 범위를 따른다. 현재 단계:1단계 구현 착수; 새 run root `/content/local_fast/lsdrag-head-v13/results/argument_control_20261015`. 기존 v2 실행은 재개하지 않고 fresh labels를 수집한다. 모델/데이터 복원 완료.

# Proposed replacement protocol — awaiting scope confirmation

Read `docs/JEV_ARGUMENT_CONTROL_PLAN_20261015.md` for the new argument-minimization proposal. Stage1 is contract/implementation/functional smoke; proposed stage2 is a fresh paired20-train-question diagnostic, with optional extension to head learning only if confirmed. No new implementation or experiment has started. Inputs and runtime are restored and verified, including Gemma text prefill and embedding text/image GPU checks. Earlier commands remain on hold.

# Execution hold — 2026-10-15 recovery

User stopped the pilot and now requests argument-generation minimization design review and agreement before revising the plan and proceeding. All commands below are historical and **must not be launched automatically**. Input/model restoration alone is currently underway. No replacement tool/argument contract has been frozen. See CURRENT.md and EXPERIMENT_STATUS.json for verified recovery state.

# Active full-head experiment

Date2026-10-15. World-model work is deferred. This pilot only tests head feasibility under the restored full tool interface.

Exact plan: `lab/registry/full_head_20261015/execution_plan_v2.json`; data and asset identities are alongside it. Astra review: `docs/JEV_FULL_HEAD_PILOT_ASTRA_REVIEW.md`.

## Paths

- Code/Git: `/content/local_fast/lsdrag-code`
- Inputs: `/content/local_fast/lsdrag-head-v12/data`
- Corpus: `/content/local_fast/lsdrag-work/data/ir_final288`
- Indexes: `/content/local_fast/lsdrag-work/indexes_final288`
- Run: `/content/local_fast/lsdrag-head-v12/results/full_head_20261015_v2`
- Logs: `/content/local_fast/lsdrag-head-v12/logs`
- Drive backup target: `/content/drive/MyDrive/lsdrag-backups/20261015/full_head_20261015`

## Execution order

1. First20 fixed training questions: actual generated teacher actions, actual arguments/tools/images/answers, raw pre-choice features. This is diagnostic collection, not head training.
2. Complete first60. If any action has fewer than5 distinct training qids, extend to exactly80, then100. Do not force rare labels or filter by teacher success.
3. Train at a completed60/80/100 prefix only; require all6 action coverage and30 distinct teacher qids. Three predeclared seeds,100epochs, linear6, plain CE. Normalize on train only and fold weights.
4. Paired20-question development comparison: generated-letter/commonargs, first-token/commonargs, all3 heads. No selection of best development seed. Actual final answers generated for every policy.
5. Report teacher behavior, coverage, imitation, closed-loop QA/failures/action distribution. If coverage fails at100, report NoGo instead of pretending a full6 head was learned.

## Commands

Run from the fast Git root with `PYTHONPATH=src`. SharedGPU is authorized when free VRAM>=32GiB; timing is uncontrolled. The common arguments below deliberately omit per-record Drive mirroring; archive locally and copy to Drive after phases.

```bash
PYTHONPATH=src python -u scripts/run_full_head_pilot.py --phase teacher --start 0 --limit 20 --model-path /content/local_fast/lsdrag-v11/hf/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2 --corpus /content/local_fast/lsdrag-work/data/ir_final288 --indexes /content/local_fast/lsdrag-work/indexes_final288 --cache-dir /content/local_fast/lsdrag-v11/hf --train-questions /content/local_fast/lsdrag-head-v12/data/train_questions.jsonl --dev-questions /content/local_fast/lsdrag-head-v12/data/dev_questions.jsonl --gold /content/local_fast/lsdrag-head-v12/data/gold.jsonl --prompt prompts/search_linked.txt --output /content/local_fast/lsdrag-head-v12/results/full_head_20261015_v2 --scorer /content/local_fast/lsdrag-v11/scorer --dataset-manifest /content/local_fast/lsdrag-head-v12/data/manifest.json
```

Then teacher `--start 20 --limit 40` to reach60; optional `--start 60 --limit 20` and `--start 80 --limit 20`. Train/evaluate use the same common arguments. Verify current artifacts before resuming; completion is `.pt` plus matching `.json`, and protocol/data/source hashes must agree.

```text
--model-path /content/local_fast/lsdrag-v11/hf/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2 --corpus /content/local_fast/lsdrag-work/data/ir_final288 --indexes /content/local_fast/lsdrag-work/indexes_final288 --cache-dir /content/local_fast/lsdrag-v11/hf --train-questions /content/local_fast/lsdrag-head-v12/data/train_questions.jsonl --dev-questions /content/local_fast/lsdrag-head-v12/data/dev_questions.jsonl --gold /content/local_fast/lsdrag-head-v12/data/gold.jsonl --prompt prompts/search_linked.txt --output /content/local_fast/lsdrag-head-v12/results/full_head_20261015_v2 --scorer /content/local_fast/lsdrag-v11/scorer --dataset-manifest /content/local_fast/lsdrag-head-v12/data/manifest.json
```

## Interpretation

Generated letters are a controlled A-RAG adaptation and not the original native serialization baseline. BC includes failed teacher decisions and does not optimize downstream correctness.20 development questions cannot establish noninferiority or paper novelty. Prior question exposure is audited; no old predictions/labels are reused and there is no strict unseen-source claim. Infrastructure errors halt; expected controller format/argument failures are preserved.

## Protocol correction

The original full_head_20261015 attempt is invalid (selector emitted native tool-call text). v2 applies a common latest-user instruction and feature-contract identity; actual model prefix probe2/2 passed. No original-attempt labels are reusable. User approved v2 staged execution after seeing the plan table.

## Sequential continuation coordinator

`scripts/continue_full_head_pilot.py --common-args lab/registry/full_head_20261015/common_args_v2.json --wait-pid 784028 --backup /content/drive/MyDrive/lsdrag-backups/20261015/full_head_20261015` was initially run in tool session77816; that session terminated and is historical. Read run/launch.json for the current detached collector/coordinator PIDs. It waits for the original first20 process, then serially runs60/80/100 as coverage requires, CPU training, and paired evaluation. It never launches a second GPU phase concurrently with its collector. The coordinator writes `coordinator_status.json`, phase logs and compact phase archives. Verify actual processes before restarting; do not start a duplicate collector. If explicitly stopping the experiment, stop the coordinator first to prevent an automatic next phase.

The first collector/coordinator received an unknown SIGTERM after12 saved questions. They were restarted in independent process groups with logs redirected to local files; valid records were retained. Current PIDs/logs are in run/launch.json. Never infer these old IDs remain live after a reset.
