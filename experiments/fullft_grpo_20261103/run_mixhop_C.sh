#!/usr/bin/env bash
# Mixed-hop arm C: reader LLM frozen, head only; identical to B1 except no LLM update.
set -u
cd /content/local_fast/lsdrag-code
export PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2
LOG=/content/local_fast/lsdrag-text-v1/logs
echo "$(date -u +%FT%TZ) start mh C1" >> $LOG/fullft_main.log
/content/local_fast/lsdrag-text-v1/runtime/bin/python scripts/run_fullft_grpo.py --arm head --freeze-llm --name mh_headonly_C1 --lr-head 5e-5 \
  --split configs/split_mixhop.json --resample --questions 192 --iterations 4 --repeats 8 --reward discount --gamma 0.8 --beta 0.02 --eval-at 0,2,4 > $LOG/mh_headonly_C1.log 2>&1
echo "$(date -u +%FT%TZ) mh C1 exit $?" >> $LOG/fullft_main.log
