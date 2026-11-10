"""Common train-only native-imitation initialization; no validation tuning."""
import json
import argparse
import copy
from pathlib import Path
import torch
from torch.nn import functional as F
from lsdrag.decision_policy import PolicyHead
from lsdrag.decision_policy_followup import state_weights
from common import ROOT, save, sha, load_head


def fit_v22(spec):
    """Fixed backbone, all used tool/pointer layers; internal-qid validation only."""
    from lsdrag.chunk_pointer import ChunkPointer
    from collections import Counter
    out = Path(spec['output'])
    out.mkdir(parents=True, exist_ok=False)
    save(out/'spec.json', spec)
    teacher = Path(spec['teacher'])
    assert json.loads((teacher/'status.json').read_text())['phase'] == 'complete'
    records = torch.load(teacher/'teacher_states.pt', weights_only=True)
    split = json.loads(Path(spec['split']).read_text())
    fit_ids, val_ids = set(split['fit']), set(split['validation'])
    assert not fit_ids & val_ids
    assert {r['qid'] for r in records} <= fit_ids | val_ids
    fit = [r for r in records if r['qid'] in fit_ids]
    val = [r for r in records if r['qid'] in val_ids]
    fit_read = [r for r in fit if r['pointer_target'] is not None]
    val_read = [r for r in val if r['pointer_target'] is not None]
    coverage = {'fit_read_qids': len({r['qid'] for r in fit_read}),
                'validation_read_qids_by_hop': {h: len({r['qid'] for r in val_read if r['qid'].startswith(h)})
                                                for h in ('2hop', '3hop', '4hop')}}
    coverage['passed'] = coverage['fit_read_qids'] >= 100 and min(coverage['validation_read_qids_by_hop'].values()) >= 10
    save(out/'coverage.json', coverage)
    if not coverage['passed']:
        save(out/'status.json', dict(phase='gate_failed', gate='pointer_coverage', pid=None))
        return
    torch.set_num_threads(2)
    torch.manual_seed(spec['seed'])
    assert sha(Path(spec['bc_head'])) == spec['bc_head_sha256']
    head = load_head({'checkpoint': spec['bc_head']})
    pointer = ChunkPointer()
    for name, p in head.named_parameters():
        p.requires_grad_(name.startswith(('trunk.', 'action.', 'skip.')))
    params = [p for p in head.parameters() if p.requires_grad] + list(pointer.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4)

    def tool_loss(rows):
        x = torch.stack([r['pack'][0:1] for r in rows]).float()
        legal = torch.stack([r['legal'] for r in rows])
        y = torch.tensor([r['action'] for r in rows])
        logp = head(x, legal)[0].log_softmax(-1)
        nll = -logp.gather(1, y[:, None]).squeeze(1)
        smooth = -logp.masked_fill(~legal, 0).sum(1)/legal.sum(1)
        w = state_weights([r['qid'] for r in rows], 'cpu')
        return (w*(.95*nll+.05*smooth)).sum(), float((logp.argmax(-1) == y).float().mean())

    def ptr_loss(rows):
        weights = state_weights([r['qid'] for r in rows], 'cpu')
        losses = []
        for r in rows:
            _, ev = pointer.select(r['pack'][0].float(), r['candidate_features'].float(), targets=r['pointer_target'])
            losses.append(-torch.stack([e['logp'] for e in ev]).mean())
        return (weights * torch.stack(losses)).sum()

    best, history = float('inf'), []
    for epoch in range(100):
        head.train(); pointer.train(); opt.zero_grad()
        tl, _ = tool_loss(fit); pl = ptr_loss(fit_read)
        loss = tl + pl
        assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.); opt.step()
        head.eval(); pointer.eval()
        with torch.no_grad():
            vt, acc = tool_loss(val); vp = ptr_loss(val_read); value = float(vt + vp)
        history.append(dict(epoch=epoch, tool_train_ce=float(tl.detach()), pointer_train_ce=float(pl.detach()),
                            validation_ce=value, tool_validation_accuracy=acc))
        if value < best:
            best = value
            torch.save({'tool': copy.deepcopy(head.state_dict()), 'pointer': copy.deepcopy(pointer.state_dict()),
                        'epoch': epoch, 'format': 'v22-tool-pointer-v1', 'seed': spec['seed']}, out/'head.pt')
        save(out/'status.json', dict(phase='training', epoch=epoch, validation_ce=value, pid=__import__('os').getpid()))
    ck = torch.load(out/'head.pt', weights_only=True)
    head.load_state_dict(ck['tool']); pointer.load_state_dict(ck['pointer']); head.eval(); pointer.eval()
    def sf1(a, b):
        a, b = set(a), set(b)
        return 2*len(a & b)/(len(a)+len(b)) if a or b else 1.
    checks = []
    with torch.no_grad():
        for r in val_read:
            pred, _ = pointer.select(r['pack'][0].float(), r['candidate_features'].float())
            teacher_indices = [r['candidate_ids'].index(cid) for cid in r['teacher_chunk_ids'] if cid in r['candidate_ids']]
            checks.append(dict(qid=r['qid'], source=r['pointer_source'], predicted=pred,
                               target=r['pointer_target'], teacher=teacher_indices,
                               pointer_support_f1=sf1(pred, r['gold_candidate_indices']),
                               teacher_support_f1=sf1(teacher_indices, r['gold_candidate_indices']),
                               exact=pred == r['pointer_target']))
    gold_checks = [c for c in checks if c['source'] == 'gold_present']
    def balanced(key):
        by_q = {}
        for c in gold_checks:
            by_q.setdefault(c['qid'], []).append(c[key])
        return sum(sum(v)/len(v) for v in by_q.values()) / len(by_q) if by_q else None
    pf, tf = balanced('pointer_support_f1'), balanced('teacher_support_f1')
    quality_pass = pf is not None and pf >= tf
    report = dict(coverage=coverage, pointer_support_f1=pf, teacher_support_f1=tf,
                  quality_gate_passed=quality_pass, gold_present_validation_states=len(gold_checks),
                  validation_checks=checks, learning_curve=history, selected_epoch=ck['epoch'],
                  head_sha256=sha(out/'head.pt'), teacher_sha256=sha(teacher/'teacher_states.pt'),
                  label_type='teacher action KD plus gold-present pointer supervision')
    save(out/'summary.json', report)
    save(out/'status.json', dict(phase='complete' if quality_pass else 'gate_failed', gate='pointer_quality', pid=None))
    print(json.dumps({k:v for k,v in report.items() if k not in ('validation_checks','learning_curve')}, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec')
    args = parser.parse_args()
    if args.spec:
        spec = json.loads(Path(args.spec).read_text())
        assert spec['profile'] == 'v22'
        return fit_v22(spec)
    run = ROOT/'results/decision_shared_teacher192_20261023'
    out = ROOT/'results/decision_shared_initialization_20261023'
    assert json.loads((run/'status.json').read_text())['phase'] == 'complete'
    spec = json.loads((run/'spec.json').read_text())
    records = torch.load(run/'teacher_states.pt', weights_only=True)
    assert {r['qid'] for r in records} == set(spec['qids'])
    torch.set_num_threads(2)
    torch.manual_seed(20261051)
    x = torch.stack([r['pack'] for r in records]).float()
    legal = torch.stack([r['legal'] for r in records])
    y = torch.tensor([r['action'] for r in records])
    weights = state_weights([r['qid'] for r in records], 'cpu')
    head = PolicyHead()
    with torch.no_grad():
        head.mean.copy_(x[:, 0].mean(0))
        head.scale.copy_(x[:, 0].std(0, unbiased=False).clamp_min(.1))
        head.value.weight.zero_()
        head.value.bias.zero_()
    for name, param in head.named_parameters():
        param.requires_grad_(name.startswith(('trunk.', 'action.', 'skip.')))
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=3e-4)
    curve = []
    for epoch in range(100):
        opt.zero_grad()
        logits = head(x, legal)[0]
        logp = F.log_softmax(logits.masked_fill(~legal, -1e9), dim=-1)
        nll = -logp.gather(1, y[:, None]).squeeze(1)
        smooth = -logp.masked_fill(~legal, 0).sum(1)/legal.sum(1)
        loss = (weights*(.95*nll+.05*smooth)).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        opt.step()
        if epoch%10==0 or epoch==99:
            curve.append(dict(epoch=epoch, loss=float(loss.detach())))
    out.mkdir(parents=True, exist_ok=False)
    torch.save(head.state_dict(), out/'head.pt')
    with torch.no_grad():
        accuracy = float((head(x, legal)[0].argmax(-1)==y).float().mean())
    save(out/'manifest.json', dict(kind='original', checkpoint=str(out/'head.pt'), sha256=sha(out/'head.pt'), train=spec['qids'], seed=20261051, epochs=100, lr=3e-4, label_smoothing=.05, teacher_states_sha256=sha(run/'teacher_states.pt'), train_accuracy=accuracy, curve=curve, note='Training fit only, not heldout quality. This common initialization precedes both GRPO and all-action improvement; no prior small-pool weights loaded.'))
    save(out/'status.json', dict(phase='complete', pid=None))

if __name__ == '__main__':
    main()
