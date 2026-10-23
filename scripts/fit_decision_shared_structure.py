"""One multi-position MLP candidate plus matched warmstart, using existing train caches."""
import copy
import json
import torch
from torch.nn import functional as F
from run_decision_policy_pilot import ROOT, REPO, save, sha
from lsdrag.decision_state_learning import StateHead, action_values, expected_utility_loss
from lsdrag.decision_policy import distribution
from lsdrag.decision_policy_followup import state_weights

def read(path):return json.loads(path.read_text())

def main():
    teacher=ROOT/'results/decision_shared_teacher192_20261023'
    outcomes=ROOT/'results/decision_shared_outcomes192_20261023'
    out=ROOT/'results/decision_shared_structure_20261023'
    spec=read(REPO/'lab/registry/decision_shared_structure_20261023/plan.json')
    assert spec['inputs']=='multi' and spec['body']=='mlp' and spec['lambda']==.03
    assert sha(REPO/'scripts/fit_decision_shared_structure.py')==spec['script_sha256']
    assert read(outcomes/'status.json')['phase']=='complete'
    assert read(teacher/'status.json')['phase']=='complete'
    labels=torch.load(teacher/'teacher_states.pt',weights_only=True)
    records=torch.load(outcomes/'states.pt',weights_only=True)
    m=read(outcomes/'manifest.json')
    assert {r['qid'] for r in labels}=={r['qid'] for r in records}==set(m['train'])
    torch.set_num_threads(2);torch.manual_seed(spec['seed'])
    x=torch.stack([r['pack'] for r in labels]).float();legal=torch.stack([r['legal'] for r in labels]);act=torch.tensor([r['action'] for r in labels]);w=state_weights([r['qid'] for r in labels],'cpu')
    h=StateHead(inputs='multi',body='mlp');h.normalize_from_training(x)
    opt=torch.optim.AdamW(h.parameters(),lr=3e-4)
    for _ in range(100):
        opt.zero_grad();lp=F.log_softmax(h(x,legal)[0],dim=-1)
        nll=-lp.gather(1,act[:,None]).squeeze(1);smooth=-lp.masked_fill(~legal,0).sum(1)/legal.sum(1)
        loss=(w*(.95*nll+.05*smooth)).sum();loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.);opt.step()
    out.mkdir(parents=True,exist_ok=False)
    torch.save(h.state_dict(),out/'multi_mlp_bc.pt')
    del x,legal,act,w
    x=torch.stack([r['pack'] for r in records]).float();legal=torch.stack([r['legal'] for r in records]);w=state_weights([r['qid'] for r in records],'cpu');q,_,_=action_values(records,.03)
    with torch.no_grad():ref=distribution(h(x,legal)[0],legal).probs.detach()
    opt=torch.optim.AdamW(h.parameters(),lr=1e-4);curve=[]
    for epoch in range(100):
        opt.zero_grad();loss=expected_utility_loss(h(x,legal)[0],legal,q,ref,w);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.);opt.step()
        if epoch%10==0 or epoch==99:curve.append(dict(epoch=epoch,loss=float(loss.detach())))
    torch.save(h.state_dict(),out/'multi_mlp_outcome.pt')
    candidates={}
    for name in ['multi_mlp_bc','multi_mlp_outcome']:
        p=out/f'{name}.pt';candidates[name]=dict(kind='structure',inputs='multi',body='mlp',checkpoint=str(p),sha256=sha(p))
    save(out/'manifest.json',dict(candidates=candidates,train=m['train'],teacher_sha256=sha(teacher/'teacher_states.pt'),outcomes_sha256=sha(outcomes/'states.pt'),plan=spec,curve=curve,note='Matched multi-MLP warmstart/outcome pair. Cross-architecture comparison also changes parameter count, trainability, KL reference, and features; not a pure single-factor ablation. Q suffix policy is frozen original BC, so improvement surrogate is off-policy relative to this warmstart. No additional LLM training rollouts.'))
    save(out/'status.json',dict(phase='complete',pid=None))

if __name__=='__main__':main()
