#!/usr/bin/env bash
# Wait until the GPU has been idle (util <= 5%) for 30 consecutive minutes, then run fresh48 agent episodes and
# the idea-4 re-answer. Never touches other processes.
set -u
cd /content/local_fast/lsdrag-code
idle_since=""
while true; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  now=$(date +%s)
  if [ "$util" -le 5 ] && [ "$free" -ge 40000 ]; then
    idle_since=${idle_since:-$now}
    if [ $((now-idle_since)) -ge 1800 ]; then break; fi
  else
    idle_since=""
  fi
  echo "$(date -u +%FT%TZ) util=$util free=$free idle_for=$(( idle_since ? now-idle_since : 0 ))s"
  sleep 30
done
echo "$(date -u +%FT%TZ) GPU idle 30 min; launching"
PYTHONPATH=src:scripts python -u scripts/evaluate_decision_shared_corpus.py --spec lab/registry/poc_idea4_fresh48_20261026/evaluation_spec.json || exit 1
PYTHONPATH=src:scripts python -u scripts/poc_idea4_fresh_eval.py
echo "$(date -u +%FT%TZ) done"
