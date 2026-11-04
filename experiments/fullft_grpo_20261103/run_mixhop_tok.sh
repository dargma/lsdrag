#!/usr/bin/env bash
# Token-constrained reruns (fast: 2 iterations, the point compared in RESULTS): B2 then A2, sequential.
# Same as B1/A1 except reward F1*0.8^calls*0.995^generated_tokens and FINAL answer capped at 60 characters.
set -u
cd /content/local_fast/lsdrag-code
export PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=2
PY=/content/local_fast/lsdrag-text-v1/runtime/bin/python; LOG=/content/local_fast/lsdrag-text-v1/logs
C="--split configs/split_mixhop.json --resample --questions 192 --iterations 2 --repeats 8 --reward discount --gamma 0.8 --token-gamma 0.995 --answer-max-chars 60 --beta 0.02 --lr 1e-6 --eval-at 0,2 --save-model-at 2"
echo "$(date -u +%FT%TZ) start B2_tok" >> $LOG/fullft_main.log
$PY scripts/run_fullft_grpo.py --arm head --name mh_head_B2_tok --lr-head 5e-5 $C > $LOG/mh_head_B2_tok.log 2>&1
echo "$(date -u +%FT%TZ) B2_tok exit $?" >> $LOG/fullft_main.log
$PY scripts/run_fullft_grpo.py --arm native --name mh_native_A2_tok --decision-temperature 4.0 $C > $LOG/mh_native_A2_tok.log 2>&1
echo "$(date -u +%FT%TZ) A2_tok exit $?" >> $LOG/fullft_main.log
