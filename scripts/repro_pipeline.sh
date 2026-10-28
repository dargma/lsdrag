#!/usr/bin/env bash
# Corpus -> teacher -> BC -> GRPO -> dev64 eval. Stages whose output is complete are skipped.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); ROOT=/content/local_fast/lsdrag-text-v1; PY=$ROOT/runtime/bin/python; LOG=$ROOT/logs; SPECS=$ROOT/specs
cd "$REPO"; mkdir -p $LOG; export PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2
done_() { [ -f "$1" ] && grep -q '"phase": "complete"' "$1"; }
stage() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG/pipeline.log; }

CORPUS=$ROOT/corpora/musique_shared_20261023
[ -f $CORPUS/documents.jsonl ] || { stage corpus; $PY scripts/build_decision_shared_corpus.py > $LOG/corpus.log 2>&1; }
grep -q '"embeddings_complete": true' $CORPUS/manifest.json || { stage index; $PY scripts/index_decision_shared_corpus.py > $LOG/index.log 2>&1; }
if ! done_ $ROOT/results/decision_shared_teacher192_20261023/status.json; then
  stage teacher; $PY scripts/repro_head_policy.py teacher
  $PY scripts/collect_decision_shared_teacher.py --spec $SPECS/teacher_spec.json > $LOG/teacher.log 2>&1
fi
done_ $ROOT/results/decision_shared_initialization_20261023/status.json || { stage bc; $PY scripts/fit_decision_shared_initialization.py > $LOG/bc.log 2>&1; }
if ! done_ $ROOT/results/decision_shared_grpo192_20261023/status.json; then
  stage grpo; $PY scripts/repro_head_policy.py grpo
  $PY scripts/run_decision_shared_grpo.py --manifest $SPECS/decision_shared_grpo192_20261023_manifest.json > $LOG/grpo.log 2>&1
fi
if ! done_ $ROOT/results/decision_shared_validation64_repro/status.json; then
  stage eval; $PY scripts/repro_head_policy.py eval
  $PY scripts/evaluate_decision_shared_corpus.py --spec $SPECS/decision_shared_validation64_repro_spec.json > $LOG/eval.log 2>&1
fi
stage complete
$PY scripts/repro_head_policy.py table --run decision_shared_validation64_repro | tee -a $LOG/pipeline.log
