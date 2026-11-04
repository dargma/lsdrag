#!/usr/bin/env bash
# Mixed-hop (2/3/4hop) full-FT GRPO: A1 native then B1 head+LLM, sequential. C1 (head only) runs beside via run_mixhop_C.sh.
set -u
cd /content/local_fast/lsdrag-code
export PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2
PY=/content/local_fast/lsdrag-text-v1/runtime/bin/python; LOG=/content/local_fast/lsdrag-text-v1/logs
COMMON="--split configs/split_mixhop.json --resample --questions 192 --iterations 4 --repeats 8 --reward discount --gamma 0.8 --beta 0.02 --lr 1e-6 --eval-at 0,2,4"
echo "$(date -u +%FT%TZ) start mh A1" >> $LOG/fullft_main.log
$PY scripts/run_fullft_grpo.py --arm native --name mh_native_A1 --decision-temperature 4.0 --save-model-at 4 $COMMON > $LOG/mh_native_A1.log 2>&1
echo "$(date -u +%FT%TZ) mh A1 exit $?" >> $LOG/fullft_main.log
$PY scripts/run_fullft_grpo.py --arm head --name mh_head_B1 --lr-head 5e-5 --save-model-at 4 $COMMON > $LOG/mh_head_B1.log 2>&1
echo "$(date -u +%FT%TZ) mh B1 exit $?" >> $LOG/fullft_main.log
