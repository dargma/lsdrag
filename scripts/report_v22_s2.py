"""Paired dev comparison of two policies from one evaluate_decision_shared_corpus run (CPU only)."""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ['em', 'f1', 'tool_calls', 'generated_tokens', 'prompt_tokens',
        'retrieved_support_recall', 'read_support_recall']


def load(run, policy):
    rows = {}
    for path in sorted((run/'episodes').glob(f'{policy}_*.json')):
        ep = json.loads(path.read_text())
        reads = [t for t in ep['trajectory'] if t['tool_name'] == 'read_chunk']
        searches = [t for t in ep['trajectory'] if t['tool_name'] != 'read_chunk']
        rows[ep['qid']] = dict({k: float(ep[k]) for k in KEYS}, errors=bool(ep['errors']),
                               empty=not ep['answer'].strip(), forced=bool(ep['forced_final']),
                               reads=len(reads), searches=len(searches),
                               chunks_read=sum(len(t['arguments']['chunk_ids']) for t in reads))
    return rows


def bootstrap(delta, seed=20261110, n=10000):
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), (n, len(delta)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--a', required=True, help='candidate policy')
    ap.add_argument('--b', required=True, help='reference policy')
    ap.add_argument('--run-b', help='run directory of the reference policy if different')
    ap.add_argument('--output')
    args = ap.parse_args()
    run = Path(args.run)
    a, b = load(run, args.a), load(Path(args.run_b) if args.run_b else run, args.b)
    assert a.keys() == b.keys() and a
    qids = sorted(a)
    out = dict(run=str(run), a=args.a, b=args.b, n=len(qids), policies={}, paired={}, by_hop={})
    for name, rows in ((args.a, a), (args.b, b)):
        out['policies'][name] = {k: float(np.mean([rows[q][k] for q in qids]))
                                 for k in KEYS + ['reads', 'searches', 'chunks_read']}
        out['policies'][name].update(
            errors=sum(rows[q]['errors'] for q in qids), empty=sum(rows[q]['empty'] for q in qids),
            forced_final=sum(rows[q]['forced'] for q in qids),
            chunks_per_read=sum(rows[q]['chunks_read'] for q in qids)/max(1, sum(rows[q]['reads'] for q in qids)))
    for k in KEYS:
        d = np.array([a[q][k]-b[q][k] for q in qids])
        out['paired'][k] = dict(mean_delta=float(d.mean()), ci95=bootstrap(d),
                                wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()),
                                ties=int((abs(d) <= 1e-9).sum()))
    both = [q for q in qids if a[q]['em'] == 1 and b[q]['em'] == 1]
    out['em_table'] = dict(both=len(both), gained=sum(a[q]['em'] == 1 and b[q]['em'] == 0 for q in qids),
                           lost=sum(a[q]['em'] == 0 and b[q]['em'] == 1 for q in qids),
                           neither=sum(a[q]['em'] == 0 and b[q]['em'] == 0 for q in qids),
                           both_delta_calls=float(np.mean([a[q]['tool_calls']-b[q]['tool_calls'] for q in both])) if both else None,
                           both_delta_tokens=float(np.mean([a[q]['generated_tokens']-b[q]['generated_tokens'] for q in both])) if both else None)
    for hop in ('2hop', '3hop', '4hop'):
        sub = [q for q in qids if q.startswith(hop)]
        out['by_hop'][hop] = {name: {k: float(np.mean([rows[q][k] for q in sub])) for k in ('em', 'f1', 'tool_calls', 'generated_tokens')}
                              for name, rows in ((args.a, a), (args.b, b))} | {'n': len(sub)}
    text = json.dumps(out, indent=2)
    if args.output:
        Path(args.output).write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
