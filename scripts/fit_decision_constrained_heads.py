"""Two predeclared token budgets; exact warmstart, train-only primal/dual fitting."""
import copy,json,time,os
from pathlib import Path
import torch
from evaluate_decision_value_heads import load_head
from lsdrag.decision_policy import distribution
from lsdrag.decision_state_learning import action_values
from lsdrag.decision_value_head import value_prediction_loss
from lsdrag.decision_constraints import constrained_loss,feasibility
from lsdrag.decision_policy_followup import state_weights
from run_decision_policy_pilot import ROOT,REPO,save,sha
NAME='decision_constrained_policy_20261024'
def read(p):return json.loads(p.read_text())
def main():
    plan=read(REPO/'lab/registry'/NAME/'plan.json')
    for p,h in plan['source_sha256'].items():assert sha(REPO/p)==h,p
    for p,h in plan['input_sha256'].items():assert sha(Path(p))==h,p
    out=ROOT/'results'/NAME;out.mkdir(exist_ok=False);save(out/'plan.json',plan)
    torch.set_num_threads(2);torch.manual_seed(plan['seed']);start=time.monotonic()
    def phase(name,**kw):save(out/'status.json',dict(phase=name,pid=None if name in ['complete','failed'] else os.getpid(),seconds=time.monotonic()-start,**kw))
    phase('loading')
    try:
        records=torch.load(plan['outcomes'],weights_only=True,map_location='cpu')
        assert {r['qid'] for r in records}==set(plan['train'])
        assert not set(plan['train'])&set(plan['validation']+plan['test'])
        x=torch.stack([r['pack'] for r in records]).float();legal=torch.stack([r['legal'] for r in records]);w=state_weights([r['qid'] for r in records],'cpu')
        _,q,cost=action_values(records,.03);g=torch.zeros_like(q)
        for i,r in enumerate(records):
            for a in range(4):
                samples=[s for s in r['samples'] if s['action']==a]
                if samples:g[i,a]=sum(s['generated'] for s in samples)/len(samples)
        anchor=load_head(plan['reference_spec']);baseline=load_head(plan['basic_grpo_spec'])
        with torch.no_grad():
            ref=distribution(anchor(x,legal)[0],legal).probs
            bp=distribution(baseline(x,legal)[0],legal).probs
            reference_q=float((w*(ref*q).sum(-1)).sum());baseline_g=float((w*(bp*g).sum(-1)).sum());reference_g=float((w*(ref*g).sum(-1)).sum())
        floor=max(0,reference_q-plan['quality_tolerance']);candidates={};feasible={}
        for ratio in plan['token_ratios']:
            name=f'budget{int(ratio*100)}';budget=ratio*baseline_g
            feasible[name]=feasibility(q,g,legal,w,budget,floor)
            save(out/'feasibility.json',feasible)
            h=copy.deepcopy(anchor);norm={k:v.clone() for k,v in h.named_buffers()}
            opt=torch.optim.AdamW(h.parameters(),lr=plan['lr']);dt=dq=0.;curve=[]
            phase('training',candidate=name)
            for epoch in range(plan['epochs']):
                opt.zero_grad();logits,_,pq,pc=h(x,legal)
                loss,eq,eg,vt,vq=constrained_loss(logits,legal,q,g,w,ref,budget,floor,dt,dq)
                aux=value_prediction_loss(pq,pc,legal,q,cost,w)
                (loss+.1*aux).backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.);opt.step()
                dt=max(0.,min(20.,dt+plan['dual_lr']*float(vt.detach())))
                dq=max(0.,min(20.,dq+plan['dual_lr']*float(vq.detach())))
                if epoch%20==0 or epoch==plan['epochs']-1:curve.append(dict(epoch=epoch,expected_f1=float(eq.detach()),expected_generated=float(eg.detach()),dual_tokens=dt,dual_quality=dq))
            assert all(torch.equal(v,norm[k]) for k,v in h.named_buffers())
            assert all(torch.isfinite(v).all() for v in h.state_dict().values())
            with torch.no_grad():
                p=distribution(h(x,legal)[0],legal).probs
                finalq=float((w*(p*q).sum(-1)).sum());finalg=float((w*(p*g).sum(-1)).sum())
            cp=out/(name+'.pt');torch.save(h.state_dict(),cp)
            candidates[name]=dict(kind='decision_value',body='mlp',checkpoint=str(cp),sha256=sha(cp),budget=budget,quality_floor=floor,train_expected_f1=finalq,train_expected_generated=finalg,train_constraints_met=bool(finalq>=floor and finalg<=budget),curve=curve)
        save(out/'manifest.json',dict(candidates=candidates,plan=plan,reference_q=reference_q,reference_generated=reference_g,basic_grpo_expected_generated=baseline_g,feasibility=feasible,note='Cache constraints only; off-policy continuation/state weighting not actual episode guarantees. Auxiliary cost output retains mixed-cost log units; token constraint uses raw cached generated counts, not that predictor.'))
        phase('complete',candidates=len(candidates))
    except BaseException as e:phase('failed',error=repr(e));raise
if __name__=='__main__':main()
