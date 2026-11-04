#!/usr/bin/env bash
# Follow-up C2 (C1 ended in alarm 1): identical to C1 but head lr 2e-4 (4x) -> is optimization strength the limiter?
# Iteration-0 eval is identical to C1 (same head, greedy), so it is copied from C1 instead of re-run.
set -u
cd /content/local_fast/lsdrag-code
export PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2
LOG=/content/local_fast/lsdrag-text-v1/logs
echo "$(date -u +%FT%TZ) start mh C2" >> $LOG/fullft_main.log
/content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/run_fullft_grpo.py --arm head --freeze-llm --name mh_headonly_C2_lr2e-4 --lr-head 2e-4 \
  --split configs/split_mixhop.json --resample --questions 192 --iterations 4 --repeats 8 --reward discount --gamma 0.8 --beta 0.02 --eval-at 2,4 > $LOG/mh_headonly_C2.log 2>&1
echo "$(date -u +%FT%TZ) mh C2 exit $?" >> $LOG/fullft_main.log
