#!/usr/bin/env bash
# Sequential reproduction pipeline (docs/REPRODUCE_HEAD_POLICY.md) plus the KL ablation.
# Each stage is skipped when its ROOT/results output already reports complete.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
ROOT=/content/local_fast/lsdrag-text-v1
PY=$ROOT/runtime/bin/python
LOG=$ROOT/logs
cd "$REPO"
export PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2

done_() { [ -f "$1" ] && grep -q '"phase": "complete"' "$1"; }
stage() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG/pipeline.log; }

if ! grep -q '"embeddings_complete": true' $ROOT/corpora/musique_shared_20261023/manifest.json; then
  stage index
  $PY scripts/index_decision_shared_corpus.py > $LOG/index.log 2>&1
  git checkout lab/registry/decision_shared_corpus_20261023/corpus_manifest.json
fi
if ! done_ $ROOT/results/decision_shared_teacher192_20261023/status.json; then
  stage teacher
  [ -f $ROOT/results/decision_state_policy_iteration_20261023/manifest.json ] || $PY scripts/repro_head_policy.py split
  $PY scripts/repro_head_policy.py teacher --max-seconds 7200
  $PY scripts/collect_decision_shared_teacher.py --spec lab/registry/repro_head_policy/teacher_spec.json > $LOG/teacher.log 2>&1
fi
if ! done_ $ROOT/results/decision_shared_initialization_20261023/status.json; then
  stage bc
  $PY scripts/fit_decision_shared_initialization.py > $LOG/bc.log 2>&1
fi
for spec in "decision_shared_grpo192_kl0 0" "decision_shared_grpo192_20261023 0.02"; do
  set -- $spec
  if ! done_ $ROOT/results/$1/status.json; then
    stage "grpo $1 kl=$2"
    $PY scripts/repro_head_policy.py grpo --name $1 --kl $2 --max-seconds 7200
    $PY scripts/run_decision_shared_grpo.py --manifest lab/registry/repro_head_policy/$1_manifest.json > $LOG/$1.log 2>&1
  fi
done
if ! done_ $ROOT/results/decision_shared_validation64_kl/status.json; then
  stage eval
  $PY scripts/repro_head_policy.py eval --name decision_shared_validation64_kl \
    --grpo decision_shared_grpo192_20261023 decision_shared_grpo192_kl0 --max-seconds 5400
  $PY scripts/evaluate_decision_shared_corpus.py --spec lab/registry/repro_head_policy/decision_shared_validation64_kl_spec.json > $LOG/eval_kl.log 2>&1
fi
stage complete
$PY scripts/repro_head_policy.py table --run decision_shared_validation64_kl | tee -a $LOG/pipeline.log
