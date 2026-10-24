"""Frozen train-cache fitting; no LLM or validation access."""
import copy,json,time,os
from pathlib import Path
import torch
from lsdrag.decision_value_head import DecisionValueHead,stopping_opportunity_loss,value_prediction_loss
from lsdrag.decision_state_learning import action_values,expected_utility_loss
from lsdrag.decision_policy import distribution
from lsdrag.decision_policy_followup import state_weights
from fit_decision_state_heads import load_head
from run_decision_policy_pilot import ROOT,REPO,save,sha
NAME='decision_opportunity_value_20261024'

def main():
    plan=json.loads((REPO/'lab/registry'/NAME/'plan.json').read_text())
    for p,h in plan['source_sha256'].items():assert sha(REPO/p)==h,p
    for p,h in plan['input_sha256'].items():assert sha(Path(p))==h,p
    out=ROOT/'results'/NAME;out.mkdir(exist_ok=False)
    save(out/'plan.json',plan);start=time.monotonic();torch.set_num_threads(2)
    def phase(name,**kw):save(out/'status.json',dict(phase=name,pid=None if name in ['complete','failed'] else os.getpid(),seconds=time.monotonic()-start,**kw))
    phase('loading')
    try:
        records=torch.load(plan['outcomes'],weights_only=True,map_location='cpu')
        assert set(r['qid'] for r in records)==set(plan['train'])
        assert not set(plan['train'])&set(plan['validation']+plan['test'])
        x=torch.stack([r['pack'] for r in records]).float();legal=torch.stack([r['legal'] for r in records])
        w=state_weights([r['qid'] for r in records],'cpu')
        utility,quality,cost=action_values(records,.03)
        # Same zero-quality safeguard for every candidate; contrasts isolate other interventions.
        utility=utility* (quality.masked_fill(~legal,0).max(-1).values>0)[:,None]
        anchor=load_head(plan['reference_spec'])
        torch.manual_seed(plan['seed']);base=DecisionValueHead(body='mlp')
        with torch.no_grad():
            base.mean.copy_(anchor.mean.reshape(3,-1));base.scale.copy_(anchor.scale.reshape(3,-1))
            base.encoder.weight.copy_(anchor.net[0].weight);base.encoder.bias.copy_(anchor.net[0].bias)
            base.policy_weight.copy_(anchor.net[2].weight);base.policy_bias.copy_(anchor.net[2].bias)
            assert torch.allclose(base(x,legal)[0],anchor(x,legal)[0],atol=2e-5,rtol=2e-5)
        bases={'mlp':base}
        labels=torch.load(plan['teacher'],weights_only=True,map_location='cpu')
        assert {r['qid'] for r in labels}==set(plan['train'])
        tx=torch.stack([r['pack'] for r in labels]).float();tl=torch.stack([r['legal'] for r in labels])
        tw=state_weights([r['qid'] for r in labels],'cpu')
        with torch.no_grad():target=distribution(anchor(tx,tl)[0],tl).probs
        torch.manual_seed(plan['seed']);attention=DecisionValueHead(body='attention')
        attention.mean.copy_(base.mean);attention.scale.copy_(base.scale)
        opt=torch.optim.AdamW(attention.parameters(),lr=3e-4)
        for _ in range(plan['warmstart_epochs']):
            opt.zero_grad();p=distribution(attention(tx,tl)[0],tl).probs
            loss=(tw*(target*(target.clamp_min(1e-9).log()-p.clamp_min(1e-9).log())).sum(-1)).sum()
            loss.backward();torch.nn.utils.clip_grad_norm_(attention.parameters(),1.);opt.step()
        bases['attention']=attention
        warmstart_kl=float(loss.detach());del tx,tl,target,labels
        candidates={}
        for name,config in plan['candidates'].items():
            phase('training',candidate=name)
            h=copy.deepcopy(bases[config['body']]);norm={k:v.clone() for k,v in h.named_buffers()}
            with torch.no_grad():ref=distribution(h(x,legal)[0],legal).probs
            opt=torch.optim.AdamW(h.parameters(),lr=plan['lr']);curve=[]
            for epoch in range(plan['epochs']):
                opt.zero_grad();logits,_,pq,pc=h(x,legal)
                policy=expected_utility_loss(logits,legal,utility,ref,w)
                stop=stopping_opportunity_loss(logits,legal,quality,cost,w)
                aux=value_prediction_loss(pq,pc,legal,quality,cost,w)
                loss=policy+config['stop_coef']*stop+config['aux_coef']*aux
                loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.);opt.step()
                if epoch%20==0 or epoch==plan['epochs']-1:curve.append(dict(epoch=epoch,policy=float(policy.detach()),stop=float(stop.detach()),aux=float(aux.detach())))
            assert all(torch.equal(v,norm[k]) for k,v in h.named_buffers())
            assert all(torch.isfinite(v).all() for v in h.state_dict().values())
            cp=out/(name+'.pt');torch.save(h.state_dict(),cp)
            with torch.no_grad():
                logits,_,pq,pc=h(x,legal);p=distribution(logits,legal).probs
                metrics={'cached_expected_utility':float((w*(p*utility).sum(-1)).sum()),'cached_argmax_changed':float((p.argmax(-1)!=ref.argmax(-1)).float().mean()),'quality_mse':float((pq-quality)[legal].square().mean()),'logcost_mse':float((pc-torch.log1p(cost))[legal].square().mean())}
            candidates[name]=dict(kind='decision_value',body=config['body'],checkpoint=str(cp),sha256=sha(cp),parameters=sum(p.numel() for p in h.parameters()),config=config,curve=curve,train_diagnostics=metrics)
            save(out/'candidates.json',candidates)
        save(out/'manifest.json',dict(candidates=candidates,plan=plan,attention_warmstart_train_kl=warmstart_kl,note='Offline fixed-BC cache; train diagnostics only, not current-policy quality or calibrated confidence. MLP starts exactly selected head; attention needs distillation and differs in capacity.'))
        phase('complete',candidates=len(candidates))
    except BaseException as e:phase('failed',error=repr(e));raise
if __name__=='__main__':main()
