"""Frozen data-diversity test: one72-question fit vs unchanged36-question critic."""
import json,time
from pathlib import Path
import torch
from probe_decision_future_critic import ROOT,REPO,sha,save,tensors,metrics
from lsdrag.decision_future_critic import FutureCritic,masked_loss
NAME='decision_future_critic_probe72_20261024'
COLLECTION='decision_h3_diversity48_20261024'
def main():
    plan=json.loads((REPO/'lab/registry'/NAME/'plan.json').read_text())
    assert not plan['pending_completed_collection_hash']
    assert all(p in plan['input_sha256'] for p in plan['inputs'])
    for p,h in plan['source_sha256'].items():assert sha(REPO/p)==h
    for p,h in plan['input_sha256'].items():assert sha(Path(p))==h
    assert json.loads((REPO/'lab/reports'/COLLECTION/'independent_verification.json').read_text())['verified']
    rows=[]
    for p in plan['inputs']:rows.extend(torch.load(p,map_location='cpu',weights_only=True))
    assert len({(r['qid'],r['turn']) for r in rows})==len(rows)
    excluded=[dict(qid=r['qid'],turn=r['turn']) for r in rows if any(s['errors'] for s in r['samples'])]
    rows=[r for r in rows if not any(s['errors'] for s in r['samples'])]
    train=[r for r in rows if r['qid'] in plan['fit']];monitor=[r for r in rows if r['qid'] in plan['monitor']]
    assert len(plan['fit'])==72 and len(plan['monitor'])==12
    assert not set(plan['fit'])&set(plan['monitor']+plan['excluded_prior_monitor'])
    torch.set_num_threads(2);torch.manual_seed(20261077)
    out=ROOT/NAME;out.mkdir(exist_ok=False);save(out/'plan.json',plan);start=time.monotonic()
    model=FutureCritic();x,m,q,c,w=tensors(train)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.01);curve=[]
    for step in range(100):
        opt.zero_grad();pq,pc,_=model(x);loss=masked_loss(pq,pc,q,c,m,w);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%20==0 or step==99:curve.append([step,float(loss.detach())])
    mx,mm,mq,mc,mw=tensors(monitor)
    old=FutureCritic();old.load_state_dict(torch.load(plan['old_checkpoint'],map_location='cpu',weights_only=True))
    with torch.no_grad():
        pq,pc,actor=model(mx);oq,oc,_=old(mx);tq,tc,_=model(x)
    assert torch.count_nonzero(actor)==0 and all(torch.isfinite(v).all() for v in model.state_dict().values())
    bw=w[:,None]*m/m.sum(-1,keepdim=True)
    bq=(torch.nan_to_num(q)*bw).sum(0)/bw.sum(0);bc=(torch.nan_to_num(c)*bw).sum(0)/bw.sum(0)
    results=dict(expanded=metrics(monitor,pq,pc),old=metrics(monitor,oq,oc),baseline=metrics(monitor,bq.expand_as(pq),bc.expand_as(pc)))
    a=results['expanded'];enough=a['informative_pairs']>=10 and a['informative_questions']>=5
    passed=enough and a['gap_sign_accuracy']>.5 and all(a[k]<results[b][k] for b in ['old','baseline'] for k in ['gap_mse','cost_mae'])
    torch.save(model.state_dict(),out/'critic.pt');reload=FutureCritic();reload.load_state_dict(torch.load(out/'critic.pt',weights_only=True))
    with torch.no_grad():assert torch.equal(reload(mx)[0],pq)
    predictions=[]
    for i,r in enumerate(monitor):
        predictions.append(dict(qid=r['qid'],turn=r['turn'],prefix_sha256=r['prefix_sha256'],branches=[dict(action=s['action'],f1=s['f1'],generated=s['generated'],predicted_f1=float(pq[i,s['action']]),predicted_generated=float(pc[i,s['action']]),old_f1=float(oq[i,s['action']]),old_generated=float(oc[i,s['action']]),baseline_f1=float(bq[s['action']]),baseline_generated=float(bc[s['action']])) for s in r['samples']]))
    result=dict(status='complete',results=results,train=metrics(train,tq,tc),coverage_sufficient=enough,gate_passed=passed,curve=curve,excluded=excluded,fit_questions=len({r['qid'] for r in train}),monitor_questions=len({r['qid'] for r in monitor}),fit_states=len(train),monitor_states=len(monitor),actor_unchanged=True,new_qa_episodes=0,checkpoint_sha256=sha(out/'critic.pt'),seconds=time.monotonic()-start)
    save(out/'summary.json',result);save(out/'predictions.json',predictions);save(out/'status.json',dict(phase='complete',pid=None))
    for name,obj in [('summary',result),('predictions',predictions)]:save(REPO/'lab/reports'/NAME/(name+'.json'),obj)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
