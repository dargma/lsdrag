#!/usr/bin/env bash
# After run_lora_head_grpo.py: dev64 eval of LoRA+head, and of head-only GRPO at the same
# trajectory count (original run, iteration 0) for a matched comparison.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); ROOT=/content/local_fast/lsdrag-text-v1; PY=$ROOT/runtime/bin/python; LOG=$ROOT/logs; SPECS=$ROOT/specs
RUN=${1:-decision_shared_lora_grpo192}
cd "$REPO"; export PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2 NO_GPU_LOCK=1
until grep -qE '"phase": "(complete|failed)"' $ROOT/results/$RUN/status.json; do sleep 60; done
grep -q '"phase": "complete"' $ROOT/results/$RUN/status.json
$PY scripts/repro_head_policy.py eval --name ${RUN}_validation64 --grpo $RUN --iteration 0 --heads-only --max-seconds 5400
LORA_PATH=$ROOT/results/$RUN/lora_iteration0.pt $PY scripts/eval_lora_head.py --spec $SPECS/${RUN}_validation64_spec.json > $LOG/${RUN}_eval.log 2>&1
$PY scripts/repro_head_policy.py eval --name decision_shared_grpo_iter0_validation64 --iteration 0 --heads-only --max-seconds 5400
$PY scripts/eval_lora_head.py --spec $SPECS/decision_shared_grpo_iter0_validation64_spec.json > $LOG/grpo_iter0_eval.log 2>&1
$PY scripts/repro_head_policy.py table --run ${RUN}_validation64 decision_shared_grpo_iter0_validation64 | tee $LOG/${RUN}_table.txt
