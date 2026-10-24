"""Fit one stopping preference on verified greedy H1 branch outcomes."""
import json
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from evaluate_decision_stop_value_h2 import load_head
from lsdrag.decision_stop_value import StopValueHead, stop_margin, supported_stop_label
from lsdrag.decision_policy_followup import state_weights
from lsdrag.decision_budget_targets import budget_targets
from run_decision_policy_pilot import ROOT, REPO, save, sha

NAME = 'decision_budget_stop_h3_20261024'


def read(path):
    return json.loads(path.read_text())


def main():
    plan = read(REPO / 'lab/registry' / NAME / 'plan.json')
    for path, expected in plan['input_sha256'].items():
        assert sha(Path(path)) == expected, path
    for path, expected in plan['source_sha256'].items():
        assert sha(REPO / path) == expected, path
    assert read(Path(plan['verification']))['verified']
    records = torch.load(plan['fresh'], map_location='cpu', weights_only=True)
    labels, target_evidence = budget_targets(records)
    assert target_evidence['changed_questions'] >= 3, 'Too few supported corrections'
    coverage = {label: len({records[i]['qid'] for i, value in labels.items() if value == label})
                for label in (0, 1)}
    assert labels == {int(k): v for k, v in plan['expected_labels'].items()}
    old = []
    for path in plan['retention_features']:
        old.extend(torch.load(path, map_location='cpu', weights_only=True))
    old = list({(r['qid'], r['prefix_sha256']): r for r in old}.values())
    assert {r['qid'] for r in old} == set(plan['train_pool'])
    assert not {r['qid'] for r in old + records} & set(plan['validation'] + plan['test'])
    seen = {(r['qid'], r['prefix_sha256']) for r in records}
    retain = [r for r in old if (r['qid'], r['prefix_sha256']) not in seen]
    retain += [r for i, r in enumerate(records) if i not in labels]
    out = ROOT / 'results' / NAME
    out.mkdir(exist_ok=False)
    save(out / 'plan.json', plan)
    start = time.monotonic()

    def phase(name, **extra):
        save(out / 'status.json', dict(phase=name, pid=None if name in ('complete', 'failed')
                                      else os.getpid(), seconds=time.monotonic()-start, **extra))

    phase('training')
    try:
        torch.set_num_threads(2)
        torch.manual_seed(plan['seed'])
        base = load_head(plan['reference_spec'])
        head = StopValueHead(base)
        original = {k: v.clone() for k, v in head.base.state_dict().items()}
        ids = list(labels)
        correction = [records[i] for i in ids]

        def tensors(rows):
            return (torch.stack([r['pack'] for r in rows]).float(),
                    torch.stack([r['legal'] for r in rows]),
                    state_weights([r['qid'] for r in rows], 'cpu'))

        cx, cm, cw = tensors(correction)
        rx, rm, rw = tensors(retain)
        with torch.no_grad():
            assert torch.equal(head(cx, cm)[0], base(cx, cm)[0])
            target = .5 * stop_margin(base, cx, cm).sigmoid() + .5 * torch.tensor(list(labels.values()))
            rt = stop_margin(base, rx, rm).sigmoid()
        optimizer = torch.optim.AdamW(head.stop.parameters(), lr=plan['lr'])
        curve = []
        for epoch in range(plan['epochs']):
            optimizer.zero_grad()
            lc = (cw * F.binary_cross_entropy_with_logits(stop_margin(head, cx, cm), target,
                                                         reduction='none')).sum()
            lr = (rw * F.binary_cross_entropy_with_logits(stop_margin(head, rx, rm), rt,
                                                         reduction='none')).sum()
            (.5 * lc + .5 * lr).backward()
            torch.nn.utils.clip_grad_norm_(head.stop.parameters(), 1.)
            optimizer.step()
            if epoch % 20 == 0 or epoch == plan['epochs'] - 1:
                curve.append(dict(epoch=epoch,correction=float(lc.detach()),retention=float(lr.detach())))
        assert all(torch.equal(v, original[k]) for k, v in head.base.state_dict().items())
        assert all(torch.isfinite(v).all() for v in head.state_dict().values())
        with torch.no_grad():
            assert torch.equal(head(cx, cm)[0][:, :3], base(cx, cm)[0][:, :3])
        checkpoint = out / 'stop.pt'
        torch.save(head.state_dict(), checkpoint)
        spec = dict(kind='stop_value_h2', checkpoint=str(checkpoint), sha256=sha(checkpoint))
        restored = load_head(spec)
        assert all(torch.equal(v, restored.state_dict()[k]) for k, v in head.state_dict().items())
        save(out / 'manifest.json', dict(candidate=spec, coverage=coverage, corrections=len(labels),
             retention_states=len(retain), trainable_parameters=sum(p.numel() for p in head.stop.parameters()),
             frozen_base_verified=True, curve=curve, target_evidence=target_evidence, plan=plan))
        phase('complete')
    except BaseException as error:
        phase('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
