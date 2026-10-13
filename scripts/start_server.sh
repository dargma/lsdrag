#!/usr/bin/env bash
set -euo pipefail
# Use an already prepared local workspace; no model downloads on server startup.
task_root="${LSDRAG_WORK_ROOT:-/content/local_fast/lsdrag-work}"
mode="${1:?usage: start_server.sh reader|parser}"
export HF_HOME="$task_root/hf"
export MINERU_HOME="$task_root/mineru"
export HF_HUB_OFFLINE=1
case "$mode" in
  reader)
    exec "$task_root/runtime/bin/vllm" serve \
      "$task_root/hf/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2" \
      --served-model-name gemma-reader --host 127.0.0.1 --port 8102 \
      --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 4096 \
      --gpu-memory-utilization "${LSDRAG_GPU_FRACTION:-0.30}" --enforce-eager \
      --enable-auto-tool-choice --tool-call-parser gemma4 \
      --limit-mm-per-prompt '{"image":8,"audio":0,"video":0}' \
      --mm-processor-cache-gb 0
    ;;
  parser)
    exec "$task_root/parser-env/bin/python" "$task_root/runtime/bin/vllm" serve \
      "$task_root/hf/models--opendatalab--MinerU2.5-Pro-2604-1.2B/snapshots/d3f5e08d073c21466bbabe21c71bb1e9c2e595da" \
      --served-model-name mineru-2604 --host 127.0.0.1 --port 8101 \
      --max-model-len 8192 --max-num-seqs 8 \
      --gpu-memory-utilization "${LSDRAG_GPU_FRACTION:-0.18}" --enforce-eager
    ;;
  *) printf 'Unknown server mode: %s\n' "$mode" >&2; exit 2 ;;
esac
