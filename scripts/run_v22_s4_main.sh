#!/bin/bash
# v2.2 S4 main run: two arms, 192 questions x 8 rollouts x 2 iterations, then dev192 at i1/i2.
# One full-FT update at a time; the other arm's rollout collection overlaps it (update ~70 GiB + collection ~10 GiB).
# Usage: run_v22_s4_main.sh <snapshot dir> <gate proof path:field>...
# Stops at the first failed step. Every step reads a frozen manifest written just before it.
set -u
S=$1; shift
PROOFS="$*"
L=/content/local_fast/lsdrag-text-v1
PY=$L/runtime/bin/python
export PYTHONPATH=$S/src:$S/scripts
cd $L
LOG=$L/logs/v22_s4_main.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }

collect() {  # arm iteration
  local m; m=$($PY $S/scripts/make_v22_s4_manifest.py collect --arm $1 --iteration $2) || return 1
  local extra=""; [ $1 = head ] && extra="--pointer"
  say "start collect $1 i$2"
  $PY -u $S/scripts/run_fullft_grpo.py --arm $1 $extra --collect-only --manifest $m > logs/v22_s4_main_$1_collect$2.log 2>&1 || return 1
  grep -q '"phase": "complete"' results/v22_s4_main_$1_collect$2/status.json || return 1
  say "done collect $1 i$2"
}
update() {  # arm iteration
  local m; m=$($PY $S/scripts/make_v22_s4_manifest.py update --arm $1 --iteration $2 --proofs $PROOFS) || return 1
  local extra=""; [ $1 = head ] && extra="--pointer"
  say "start update $1 i$2"
  $PY -u $S/scripts/run_fullft_grpo.py --arm $1 $extra --update-only --manifest $m > logs/v22_s4_main_$1_update$2.log 2>&1 || return 1
  grep -q '"phase": "complete"' results/v22_s4_main_$1_update$2/status.json || return 1
  say "done update $1 i$2"
}
evaluate() {  # arm iteration
  local m; m=$($PY $S/scripts/make_v22_s4_manifest.py eval --arm $1 --iteration $2) || return 1
  say "start dev192 $1 i$2"
  $PY -u $S/scripts/evaluate_decision_shared_corpus.py --spec $m > logs/v22_s4_main_$1_dev192_i$2.log 2>&1 || return 1
  say "done dev192 $1 i$2"
}
both() {  # run two steps concurrently, fail if either fails
  "${@:1:3}" & local a=$!
  "${@:4:3}" & local b=$!
  wait $a; local ra=$?; wait $b; local rb=$?
  [ $ra -eq 0 ] && [ $rb -eq 0 ]
}
fail() { say "FAILED at: $*"; exit 1; }

say "S4 main start, snapshot $S, proofs: $PROOFS"
collect head 1 || fail "collect head 1"
both update head 1 collect native 1 || fail "update head 1 / collect native 1"
both update native 1 collect head 2 || fail "update native 1 / collect head 2"
both update head 2 collect native 2 || fail "update head 2 / collect native 2"
update native 2 & U=$!
evaluate head 1 && evaluate head 2; E=$?
wait $U || fail "update native 2"
[ $E -eq 0 ] || fail "dev192 head"
evaluate native 1 || fail "dev192 native 1"
evaluate native 2 || fail "dev192 native 2"
say "S4 main complete"
