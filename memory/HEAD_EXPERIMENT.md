# Active full-head experiment

Date2026-10-15. World-model work is deferred. This pilot only tests head feasibility under the restored full tool interface.

Exact plan: `lab/registry/full_head_20261015/execution_plan.json`; data and asset identities are alongside it. Astra review: `docs/JEV_FULL_HEAD_PILOT_ASTRA_REVIEW.md`.

## Paths

- Code/Git: `/content/local_fast/lsdrag-code`
- Inputs: `/content/local_fast/lsdrag-head-v12/data`
- Corpus: `/content/local_fast/lsdrag-work/data/ir_final288`
- Indexes: `/content/local_fast/lsdrag-work/indexes_final288`
- Run: `/content/local_fast/lsdrag-head-v12/results/full_head_20261015`
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
PYTHONPATH=src python -u scripts/run_full_head_pilot.py --phase teacher --start 0 --limit 20 --model-path /content/local_fast/lsdrag-v11/hf/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2 --corpus /content/local_fast/lsdrag-work/data/ir_final288 --indexes /content/local_fast/lsdrag-work/indexes_final288 --cache-dir /content/local_fast/lsdrag-v11/hf --train-questions /content/local_fast/lsdrag-head-v12/data/train_questions.jsonl --dev-questions /content/local_fast/lsdrag-head-v12/data/dev_questions.jsonl --gold /content/local_fast/lsdrag-head-v12/data/gold.jsonl --prompt prompts/search_linked.txt --output /content/local_fast/lsdrag-head-v12/results/full_head_20261015 --scorer /content/local_fast/lsdrag-v11/scorer --dataset-manifest /content/local_fast/lsdrag-head-v12/data/manifest.json
```

Then teacher `--start20 --limit40` to reach60; optional `--start60 --limit20` and `--start80 --limit20`. Train/evaluate use the same common arguments. Verify current artifacts before resuming; completion is `.pt` plus matching `.json`, and protocol/data/source hashes must agree.

```text
--model-path /content/local_fast/lsdrag-v11/hf/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2 --corpus /content/local_fast/lsdrag-work/data/ir_final288 --indexes /content/local_fast/lsdrag-work/indexes_final288 --cache-dir /content/local_fast/lsdrag-v11/hf --train-questions /content/local_fast/lsdrag-head-v12/data/train_questions.jsonl --dev-questions /content/local_fast/lsdrag-head-v12/data/dev_questions.jsonl --gold /content/local_fast/lsdrag-head-v12/data/gold.jsonl --prompt prompts/search_linked.txt --output /content/local_fast/lsdrag-head-v12/results/full_head_20261015 --scorer /content/local_fast/lsdrag-v11/scorer --dataset-manifest /content/local_fast/lsdrag-head-v12/data/manifest.json
```

## Interpretation

Generated letters are a controlled A-RAG adaptation and not the original native serialization baseline. BC includes failed teacher decisions and does not optimize downstream correctness.20 development questions cannot establish noninferiority or paper novelty. Prior question exposure is audited; no old predictions/labels are reused and there is no strict unseen-source claim. Infrastructure errors halt; expected controller format/argument failures are preserved.
