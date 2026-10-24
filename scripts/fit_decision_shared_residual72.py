"""One fixed outcome-supervised shared representation/actor fit."""
import json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from lsdrag.decision_shared_residual import SharedResidualPolicy
from lsdrag.decision_current_policy_targets import current_policy_targets
from probe_decision_future_critic import ROOT,REPO,sha,save,weights
from evaluate_decision_stop_value_h2 import load_head
from evaluate_decision_shared_residual import load_head as reload_head
NAME='decision_shared_residual72_20261024'
def main():
    p=json.loads((REPO/'lab/registry'/NAME/'plan.json').read_text())
    for path,h in p['hashes'].items():assert sha(Path(path))==h
    torch.set_num_threads(2);torch.manual_seed(20261079);start=time.monotonic()
    fresh=[]
    for path in p['inputs']:fresh.extend(torch.load(path,map_location='cpu',weights_only=True))
    fresh=[r for r in fresh if r['qid'] in p['fit']]
    labels,evidence=current_policy_targets(fresh);assert evidence['changed_questions']>=3
    old=[]
    for path in p['retention_features']:old.extend(torch.load(path,map_location='cpu',weights_only=True))
    old=[r for r in old if r['qid'] in p['fit']]
    seen={(r['qid'],r['prefix_sha256']) for r in fresh}
    old=list({(r['qid'],r['prefix_sha256']):r for r in old if (r['qid'],r['prefix_sha256']) not in seen}.values())
    retain=old+[r for i,r in enumerate(fresh) if i not in labels];correction=[fresh[i] for i in labels]
    assert not {r['qid'] for r in retain+correction}&set(p['excluded'])
    out=ROOT/NAME;out.mkdir(exist_ok=False);save(out/'plan.json',p)
    base=load_head(p['reference_spec']);head=SharedResidualPolicy(base)
    frozen={k:v.clone() for k,v in head.anchor.state_dict().items()}
    def tensors(rows):return torch.stack([r['pack'] for r in rows]).float(),torch.stack([r['legal'] for r in rows]),weights(rows)
    cx,cm,cw=tensors(correction);rx,rm,rw=tensors(retain)
    with torch.no_grad():
        assert torch.equal(head(cx,cm)[0],base(cx,cm)[0])
        target=.5*base(cx,cm)[0].softmax(-1)+.5*F.one_hot(torch.tensor(list(labels.values())),4)
        rt=base(rx,rm)[0].softmax(-1)
    opt=torch.optim.AdamW([v for v in head.parameters() if v.requires_grad],lr=.001,weight_decay=.01);curve=[]
    for step in range(100):
        opt.zero_grad();lc=-(cw*(target*head(cx,cm)[0].log_softmax(-1)).sum(-1)).sum();lr=-(rw*(rt*head(rx,rm)[0].log_softmax(-1)).sum(-1)).sum();loss=.5*(lc+lr);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
        if step%20==0 or step==99:curve.append([step,float(lc.detach()),float(lr.detach())])
    assert all(torch.equal(v,frozen[k]) for k,v in head.anchor.state_dict().items())
    torch.save(head.state_dict(),out/'proposal.pt')
    x=torch.cat([cx,rx]);m=torch.cat([cm,rm]);w=weights(correction+retain);trials=[];accepted=False
    with torch.no_grad():
        ref=base(x,m)[0];rp=ref.softmax(-1);rlog=ref.log_softmax(-1)
        for i in range(11):
            head.strength.fill_(2.**-i);logits=head(x,m)[0]
            kl=float((w*(rp*(rlog-logits.log_softmax(-1))).sum(-1)).sum())
            changed=float((rw*(logits[len(correction):].argmax(-1)!=ref[len(correction):].argmax(-1))).sum())
            trials.append(dict(alpha=float(head.strength),kl=kl,unsupported_change=changed))
            if kl<=.01 and changed<=.02:accepted=True;break
        if not accepted:head.strength.zero_()
        chosen=head(cx,cm)[0].argmax(-1);matches=chosen==torch.tensor(list(labels.values()));changedq={correction[i]['qid'] for i in range(len(correction)) if matches[i]}
    progress=accepted and len(changedq)>=3
    torch.save(head.state_dict(),out/'action.pt');spec=dict(kind='shared_residual',checkpoint=str(out/'action.pt'),sha256=sha(out/'action.pt'))
    restored=reload_head(spec);assert all(torch.equal(v,restored.state_dict()[k]) for k,v in head.state_dict().items())
    result=dict(candidate=spec,learning_progress=progress,supported_matches=int(matches.sum()),changed_questions=len(changedq),trials=trials,accepted=accepted,curve=curve,corrections=len(correction),retention=len(retain),target_evidence=evidence,anchor_unchanged=True,trainable_parameters=sum(v.numel() for v in head.parameters() if v.requires_grad),seconds=time.monotonic()-start)
    save(out/'summary.json',result);save(out/'status.json',dict(phase='complete',pid=None));save(REPO/'lab/reports'/NAME/'summary.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
