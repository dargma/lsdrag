"""Common train-only native-imitation initialization; no validation tuning."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from lsdrag.decision_policy import PolicyHead
from lsdrag.decision_policy_followup import state_weights
from common import ROOT, save, sha

def main():
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
