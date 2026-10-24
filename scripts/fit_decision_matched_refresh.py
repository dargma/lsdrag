"""Same-prefix 2x2 offline target-source and adaptation comparison."""
import copy
import json
import os
import time
from pathlib import Path
import torch
from lsdrag.decision_attention_adapter import AttentionAdapter
from lsdrag.decision_state_learning import action_values, expected_utility_loss
from lsdrag.decision_policy_followup import state_weights
from evaluate_decision_attention import load_head
from run_decision_policy_pilot import ROOT, REPO, save, sha

NAME = 'decision_matched_refresh_20261024'


def read(path):
    return json.loads(path.read_text())


def main():
    plan = read(REPO / 'lab/registry' / NAME / 'plan.json')
    for filename, digest in plan['source_sha256'].items():
        assert sha(REPO / filename) == digest, filename
    fresh_run = ROOT / 'results/decision_selected_refresh48_20261024'
    assert read(fresh_run / 'status.json')['phase'] == 'complete'
    audit = read(REPO / 'lab/reports/decision_selected_refresh48_20261024/audit.json')
    assert audit['audit_passed']
    fresh_path = fresh_run / 'states.pt'
    assert sha(fresh_path) == audit['states_sha256']
    old_path = Path(plan['old_states'])
    assert sha(old_path) == plan['old_states_sha256']
    assert sha(Path(plan['reference_spec']['checkpoint'])) == plan['reference_spec']['sha256']
    old = torch.load(old_path, weights_only=True, map_location='cpu')
    fresh = torch.load(fresh_path, weights_only=True, map_location='cpu')
    old_by = {(r['qid'], r['prefix_sha256']): r for r in old}
    assert len(old_by) == len(old)
    fresh = [r for r in fresh if (r['qid'], r['prefix_sha256']) in old_by]
    old = [old_by[r['qid'], r['prefix_sha256']] for r in fresh]
    # Require all predeclared questions; never quietly select successful questions.
    assert {r['qid'] for r in fresh} == set(plan['train'])
    assert not set(plan['train']) & set(plan['validation'] + plan['test'])
    for a, b in zip(old, fresh):
        assert torch.equal(a['legal'], b['legal'])
        for record in (a, b):
            for action in range(4):
                assert len([s for s in record['samples'] if s['action'] == action]) == (2 if record['legal'][action] else 0)
    out = ROOT / 'results' / NAME
    out.mkdir(exist_ok=False)
    save(out / 'plan.json', plan)
    save(out / 'input_binding.json', dict(fresh_sha256=sha(fresh_path), old_sha256=sha(old_path),
         pairs=[dict(qid=r['qid'], prefix_sha256=r['prefix_sha256']) for r in fresh],
         features='Fresh FP32 pack used identically in all four arms; no old/new feature precision confound.'))
    start = time.monotonic()
    torch.set_num_threads(2)

    def phase(name, **kwargs):
        save(out / 'status.json', dict(phase=name, pid=None if name in ('complete', 'failed') else os.getpid(),
             seconds=time.monotonic() - start, **kwargs))
    phase('loading')
    try:
        x = torch.stack([r['pack'] for r in fresh]).float()
        legal = torch.stack([r['legal'] for r in fresh])
        weights = state_weights([r['qid'] for r in fresh], 'cpu')
        base = load_head(plan['reference_spec'])
        reference = base(x, legal)[0].softmax(-1).detach()
        candidates, audits = {}, {}
        for source, records in [('old', old), ('fresh', fresh)]:
            q, _, _ = action_values(records, .03)
            for body in ('mlp', 'question_attention'):
                name = source + '_' + body
                phase('training', candidate=name)
                torch.manual_seed(plan['seed'])
                head = copy.deepcopy(base) if body == 'mlp' else AttentionAdapter(base, body)
                if body != 'mlp':
                    for parameter in head.base.parameters():
                        parameter.requires_grad_(False)
                parameters = [p for p in head.parameters() if p.requires_grad]
                buffers = {k: v.clone() for k, v in head.named_buffers()}
                with torch.no_grad():
                    assert torch.equal(head(x, legal)[0], base(x, legal)[0])
                optimizer = torch.optim.AdamW(parameters, lr=1e-4)
                curve = []
                for epoch in range(100):
                    optimizer.zero_grad()
                    loss = expected_utility_loss(head(x, legal)[0], legal, q, reference, weights)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1)
                    optimizer.step()
                    if epoch % 20 == 0 or epoch == 99:
                        curve.append(dict(epoch=epoch, loss=float(loss.detach())))
                assert all(torch.equal(value, buffers[k]) for k, value in head.named_buffers())
                assert all(torch.isfinite(value).all() for value in head.state_dict().values())
                if body != 'mlp':
                    assert all(torch.equal(value, base.state_dict()[k]) for k, value in head.base.state_dict().items())
                checkpoint = out / (name + '.pt')
                torch.save(head.state_dict(), checkpoint)
                kind = dict(kind='structure', inputs='multi', body='mlp') if body == 'mlp' else dict(kind='attention_adapter', variant=body)
                candidates[name] = dict(**kind, checkpoint=str(checkpoint), sha256=sha(checkpoint))
                audits[name] = dict(curve=curve, parameters=sum(p.numel() for p in parameters), initial_exact=True, normalization_fixed=True)
        save(out / 'manifest.json', dict(plan=plan, candidates=candidates, audits=audits, matched_states=len(fresh), questions=len(plan['train']), new_llm_rollouts=0))
        phase('complete')
    except BaseException as exc:
        phase('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
