"""Fit one four-action policy improvement on verified current-H3 terminal outcomes."""
import copy
import json
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from evaluate_decision_stop_value_h2 import load_head
from lsdrag.decision_stop_value import StopValueHead, stop_margin, supported_stop_label
from lsdrag.decision_policy_followup import state_weights
from lsdrag.decision_current_policy_targets import current_policy_targets
from lsdrag.decision_conservative_update import conservative_output_update
from run_decision_policy_pilot import ROOT, REPO, save, sha

NAME = 'decision_h3_conservative48_20261024'


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
    labels, target_evidence = current_policy_targets(records)
    assert target_evidence['changed_questions'] >= 3, 'Too few supported corrections'
    coverage = {label: len({records[i]['qid'] for i, value in labels.items() if value == label})
                for label in range(4)}
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
        head = copy.deepcopy(base)
        head.requires_grad_(False)
        head.base.net[2].requires_grad_(True)
        original = {k: v.clone() for k, v in head.state_dict().items()}
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
            target = .5 * base(cx, cm)[0].softmax(-1) + .5 * F.one_hot(torch.tensor(list(labels.values())),4).float()
            rt = base(rx, rm)[0].softmax(-1)
        optimizer = torch.optim.AdamW(head.base.net[2].parameters(), lr=plan['lr'])
        curve = []
        for epoch in range(plan['epochs']):
            optimizer.zero_grad()
            lc = (cw * -(target * head(cx, cm)[0].log_softmax(-1)).sum(-1)).sum()
            lr = (rw * -(rt * head(rx, rm)[0].log_softmax(-1)).sum(-1)).sum()
            (.5 * lc + .5 * lr).backward()
            torch.nn.utils.clip_grad_norm_(head.base.net[2].parameters(), 1.)
            optimizer.step()
            if epoch % 20 == 0 or epoch == plan['epochs'] - 1:
                curve.append(dict(epoch=epoch,correction=float(lc.detach()),retention=float(lr.detach())))
        torch.save(head.state_dict(),out/'proposal.pt')
        all_x=torch.cat((cx,rx));all_legal=torch.cat((cm,rm))
        all_qids=[r['qid'] for r in correction+retain]
        retention_mask=torch.tensor([False]*len(correction)+[True]*len(retain))
        movement=conservative_output_update(head,base,all_x,all_legal,all_qids,retention_mask,
            kl_limit=plan['kl_limit'],change_limit=plan['unsupported_change_limit'],max_backtracks=10)
        with torch.no_grad():
            matched=int((head(cx,cm)[0].argmax(-1)==torch.tensor(list(labels.values()))).sum())
        learning_progress=movement['accepted'] and matched>0
        assert all(torch.equal(v, original[k]) for k, v in head.state_dict().items() if not k.startswith('base.net.2.'))
        assert all(torch.isfinite(v).all() for v in head.state_dict().values())
        checkpoint = out / 'action.pt'
        torch.save(head.state_dict(), checkpoint)
        spec = dict(kind='stop_value_h2', checkpoint=str(checkpoint), sha256=sha(checkpoint))
        restored = load_head(spec)
        assert all(torch.equal(v, restored.state_dict()[k]) for k, v in head.state_dict().items())
        save(out / 'manifest.json', dict(candidate=spec, coverage=coverage, corrections=len(labels),
             retention_states=len(retain), trainable_parameters=sum(p.numel() for p in head.base.net[2].parameters()),
             frozen_representation_and_H3_stop_verified=True, curve=curve, target_evidence=target_evidence, movement=movement, supported_target_matches=matched, learning_progress=learning_progress, plan=plan))
        phase('complete')
    except BaseException as error:
        phase('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
