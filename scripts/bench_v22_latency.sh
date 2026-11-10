#!/bin/bash
# Controlled latency check of the final checkpoints (H-RL i2 vs N-RL i2) on a small dev subset.
# Runs four evaluator processes in ABBA order (head, native, native, head) on the same questions so
# that slow drift of the shared machine affects both arms equally, then reports paired seconds.
# Refuses to start unless the GPU has no other compute process (measurement would be meaningless).
# Usage: bench_v22_latency.sh [questions_per_hop=16]
set -eu
K=${1:-16}
L=/content/local_fast/lsdrag-text-v1
S=$L/source_snapshots/v22_s4main_bf9a67d
PY=$L/runtime/bin/python
export PYTHONPATH=$S/src:$S/scripts
cd $L
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "GPU busy, not starting:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; exit 2
fi
OUT=$L/results/v22_latency_bench_$(date -u +%Y%m%dT%H%M)
mkdir -p $OUT
for arm in head native; do $PY $S/scripts/make_v22_s4_manifest.py eval --arm $arm --iteration 2 > /dev/null; done
$PY - "$K" "$OUT" <<'E'
import json, sys
from pathlib import Path
k, out = int(sys.argv[1]), Path(sys.argv[2])
specs = Path('/content/local_fast/lsdrag-text-v1/specs/v22/s4')
for arm in ('head', 'native'):
    d = json.loads((specs/f'main_{arm}_eval2.json').read_text())
    qids = [q for h in ('2hop', '3hop', '4hop') for q in [x for x in d['qids'] if x.startswith(h)][:k]]
    for rep in ('a', 'b'):
        e = dict(d, qids=qids, run_name=f'{out.name}_{arm}_{rep}')
        (out/f'{arm}_{rep}.json').write_text(json.dumps(e, indent=2))
E
for run in head_a native_a native_b head_b; do
  echo "$(date -u +%T) start $run"
  $PY -u $S/scripts/evaluate_decision_shared_corpus.py --spec $OUT/$run.json > $OUT/$run.log 2>&1
done
mkdir -p $OUT/runs && mv $L/results/$(basename $OUT)_* $OUT/runs/
$PY - "$OUT" <<'E'
import glob, json, sys
from pathlib import Path
import numpy as np
out = Path(sys.argv[1])
def load(run):
    return {json.load(open(p))['qid']: json.load(open(p)) for p in glob.glob(f'{out}/runs/{out.name}_{run}/episodes/*.json')}
runs = {r: load(r) for r in ('head_a', 'head_b', 'native_a', 'native_b')}
qids = sorted(runs['head_a'])
sec = {arm: np.array([np.mean([runs[f'{arm}_{r}'][q]['seconds'] for r in 'ab']) for q in qids]) for arm in ('head', 'native')}
tok = {arm: np.array([runs[f'{arm}_a'][q]['generated_tokens'] for q in qids]) for arm in ('head', 'native')}
same = {arm: sum(runs[f'{arm}_a'][q]['answer'] == runs[f'{arm}_b'][q]['answer'] for q in qids) for arm in ('head', 'native')}
d = sec['head']-sec['native']
bs = d[np.random.default_rng(20261110).integers(0, len(d), (10000, len(d)))].mean(1)
res = dict(questions=len(qids), seconds_per_question={a: float(v.mean()) for a, v in sec.items()},
           generated_tokens={a: float(v.mean()) for a, v in tok.items()},
           head_minus_native_seconds=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           relative=float(d.mean()/sec['native'].mean()), head_faster_questions=int((d < 0).sum()),
           greedy_answer_repeatability=same, order='ABBA: head_a, native_a, native_b, head_b')
(out/'latency.json').write_text(json.dumps(res, indent=2)+'\n')
print(json.dumps(res, indent=2))
E
