"""One frozen train-only empirical Pareto preference pilot."""
import copy,json,time,os
from pathlib import Path
import torch
from torch.nn import functional as F
from evaluate_decision_value_heads import load_head
from lsdrag.decision_pareto import cheaper_quality_pairs
from lsdrag.decision_policy_followup import state_weights
from run_decision_policy_pilot import ROOT,REPO,save,sha
NAME='decision_pareto_refresh_20261024'
def main():
    plan=json.loads((REPO/'lab/registry'/NAME/'plan.json').read_text())
    for p,h in plan['source_sha256'].items():assert sha(REPO/p)==h,p
    for p,h in plan['input_sha256'].items():assert sha(Path(p))==h,p
    out=ROOT/'results'/NAME;out.mkdir(exist_ok=False);save(out/'plan.json',plan)
    start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(plan['seed'])
    save(out/'status.json',dict(phase='training',pid=os.getpid()))
    try:
        fresh=torch.load(plan['fresh'],map_location='cpu',weights_only=True)
        old=torch.load(plan['retention'],map_location='cpu',weights_only=True)
        assert {r['qid'] for r in old}==set(plan['train'])
        assert {r['qid'] for r in fresh}<=set(plan['train'])
        assert not set(plan['train'])&set(plan['validation']+plan['test'])
        pairs=cheaper_quality_pairs(fresh);assert len(pairs)>=10
        x=torch.stack([r['pack'] for r in fresh]).float();mask=torch.stack([r['legal'] for r in fresh])
        rx=torch.stack([r['pack'] for r in old]).float();rm=torch.stack([r['legal'] for r in old])
        w=state_weights([r['qid'] for r in old],'cpu')
        fw=state_weights([r['qid'] for r in fresh],'cpu')
        index,a,b=map(torch.tensor,zip(*pairs))
        paircounts=torch.bincount(index,minlength=len(fresh)).clamp_min(1)
        pw=fw[index]/paircounts[index];pw=pw/pw.sum()
        anchor=load_head(plan['reference_spec']);h=copy.deepcopy(anchor)
        for name,p in h.named_parameters():p.requires_grad_(name in ['encoder.weight','encoder.bias','policy_weight','policy_bias'])
        with torch.no_grad():
            ref=anchor(rx,rm)[0].softmax(-1);fref=anchor(x,mask)[0].softmax(-1)
        opt=torch.optim.AdamW([p for p in h.parameters() if p.requires_grad],lr=plan['lr'])
        curve=[]
        for epoch in range(plan['epochs']):
            opt.zero_grad();logits=h(x,mask)[0];rl=h(rx,rm)[0]
            rank=(pw*F.softplus(-(logits[index,a]-logits[index,b]))).sum()
            kl=(w*(ref*(ref.clamp_min(1e-12).log()-rl.log_softmax(-1))).sum(-1)).sum()
            fkl=(fw*(fref*(fref.clamp_min(1e-12).log()-logits.log_softmax(-1))).sum(-1)).sum()
            loss=rank+plan['kl_weight']*(kl+fkl)/2
            loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1);opt.step()
            if epoch%20==0 or epoch==plan['epochs']-1:curve.append(dict(epoch=epoch,rank=float(rank.detach()),kl=float(kl.detach()),fresh_kl=float(fkl.detach())))
        for k,v in h.state_dict().items():
            assert torch.isfinite(v).all()
            if k not in ['encoder.weight','encoder.bias','policy_weight','policy_bias']:assert torch.equal(v,anchor.state_dict()[k]),k
        cp=out/'pareto.pt';torch.save(h.state_dict(),cp)
        candidate=dict(kind='decision_value',body='mlp',checkpoint=str(cp),sha256=sha(cp))
        save(out/'manifest.json',dict(candidate=candidate,pairs=len(pairs),states=len(set(index.tolist())),curve=curve,plan=plan))
        save(out/'pairs.json',dict(pairs=pairs,note='Train-only empirical sample separation; not statistical dominance or RL.'))
        save(out/'status.json',dict(phase='complete',pid=None,seconds=time.monotonic()-start))
    except BaseException as exc:
        save(out/'status.json',dict(phase='failed',pid=None,error=repr(exc)));raise
if __name__=='__main__':main()
