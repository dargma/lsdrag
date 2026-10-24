"""Train-only opportunity counts after a completed, audited branch collection."""
import argparse
import collections
import json
from pathlib import Path

import torch

from evaluate_decision_value_heads import load_head
from run_decision_policy_pilot import ROOT, REPO, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = ROOT / 'results' / args.run
    read = lambda p: json.loads(p.read_text())
    manifest = read(run / 'manifest.json')
    assert read(run / 'status.json')['phase'] == 'complete'
    audit = read(REPO / 'lab/reports' / args.run / 'audit.json')
    assert audit['audit_passed'] and audit['states_sha256'] == sha(run / 'states.pt')
    spec = manifest['reference_spec']
    assert sha(Path(spec['checkpoint'])) == spec['sha256']
    records = torch.load(run / 'states.pt', weights_only=True, map_location='cpu')
    assert {r['qid'] for r in records} == set(manifest['train'])
    assert not set(manifest['train']) & set(manifest['validation'] + manifest['test'])
    torch.set_num_threads(2)
    head = load_head(spec, 'cpu').eval()
    with torch.no_grad():
        probabilities = head(torch.stack([r['pack'] for r in records]).float(),
                             torch.stack([r['legal'] for r in records]))[0].softmax(-1)
    pairs = []
    excluded_errors = 0
    for i, (record, probs) in enumerate(zip(records, probabilities)):
        anchor = int(probs.argmax())
        by = {a: [s for s in record['samples'] if s['action'] == a] for a in range(4)}
        bs = by[anchor]
        assert len(bs) == 2
        for action, samples in by.items():
            if action == anchor or not samples:
                continue
            assert len(samples) == 2
            if any(s['errors'] for s in bs + samples):
                excluded_errors += 1
                continue
            gap = min(s['f1'] for s in samples) - max(s['f1'] for s in bs)
            saving = min(s['generated'] for s in bs) - max(s['generated'] for s in samples)
            positive = min(s['f1'] for s in samples) > 0
            categories = []
            if gap > 0:
                categories.append('strict_quality_gain')
            if gap >= 0 and positive and saving > 0:
                categories.append('nonworse_positive_cheaper')
            if max(s['f1'] for s in bs + samples) == 0 and saving > 0:
                categories.append('zero_quality_cheaper_not_training_target')
            if not categories:
                continue
            direction = ('FINAL_to_continue' if anchor == 3 else
                         'continue_to_FINAL' if action == 3 else 'between_tools')
            pairs.append(dict(index=i, qid=record['qid'], turn=record['turn'],
                              from_action=anchor, to_action=action, direction=direction,
                              categories=categories, worst_sample_quality_gap=gap,
                              worst_sample_token_saving=saving,
                              anchor_probability=float(probs[anchor]),
                              candidate_probability=float(probs[action])))
    categories = sorted({c for p in pairs for c in p['categories']})
    summary = {}
    for category in categories:
        selected = [p for p in pairs if category in p['categories']]
        summary[category] = dict(pairs=len(selected),
                                states=len({p['index'] for p in selected}),
                                questions=len({p['qid'] for p in selected}),
                                directions=dict(collections.Counter(p['direction'] for p in selected)))
    result = dict(run=args.run, reference_spec=spec, states_sha256=audit['states_sha256'],
                  summary=summary, pairs=pairs, error_pairs_excluded=excluded_errors,
                  limitations='Training suffix evidence only. Two samples are not confidence bounds; categories may overlap. No heldout quality claim or fitted policy. Zero-quality cheaper pairs are descriptive, never correction targets.')
    save(REPO / 'lab/reports' / args.run / 'opportunity_categories.json', result)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
