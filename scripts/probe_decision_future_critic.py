"""One preregistered critic feasibility probe; never executes or updates the policy."""
import hashlib,json,time
from collections import Counter
from pathlib import Path
import torch
from lsdrag.decision_future_critic import FutureCritic,masked_loss
ROOT=Path('/content/local_fast/lsdrag-text-v1/results')
REPO=Path('/content/local_fast/lsdrag-code')
NAME='decision_future_critic_probe48_20261024'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def weights(rows):
    counts=Counter(r['qid'] for r in rows)
    return torch.tensor([1/len(counts)/counts[r['qid']] for r in rows])
def tensors(rows):
    x=torch.stack([r['pack'] for r in rows]);m=torch.stack([r['legal'] for r in rows]);q=torch.full(m.shape,float('nan'));c=q.clone()
    for i,r in enumerate(rows):
        assert {s['action'] for s in r['samples']}==set(torch.where(m[i])[0].tolist())
        for s in r['samples']:q[i,s['action']]=s['f1'];c[i,s['action']]=s['generated']
    return x,m,q,c,weights(rows)
def metrics(rows,pq,pc):
    _,m,q,c,w=tensors(rows);gap=[];sign=[];informative=set();pairs=0
    for i,r in enumerate(rows):
        errors=[];correct=[]
        acts=torch.where(m[i])[0].tolist()
        for ia,a in enumerate(acts):
            for b in acts[ia+1:]:
                truth=float(q[i,a]-q[i,b]);pred=float(pq[i,a]-pq[i,b]);errors.append((truth-pred)**2)
                if abs(truth)>1e-8:
                    pairs+=1;informative.add(r['qid']);correct.append(float(truth*pred>0))
        gap.append(sum(errors)/len(errors));sign.append(None if not correct else sum(correct)/len(correct))
    byq={}
    for r,s in zip(rows,sign):
        if s is not None:byq.setdefault(r['qid'],[]).append(s)
    return dict(gap_mse=float((torch.tensor(gap)*w).sum()),cost_mae=float((torch.where(m,(pc-c).abs(),0).sum(-1)/m.sum(-1)*w).sum()),gap_sign_accuracy=sum(sum(v)/len(v) for v in byq.values())/len(byq) if byq else None,informative_pairs=pairs,informative_questions=len(informative))
def main():
    torch.set_num_threads(2);torch.manual_seed(20261077)
    planpath=REPO/'lab/registry'/NAME/'plan.json';plan=json.loads(planpath.read_text())
    for name,h in plan['hashes'].items():assert sha(Path(name))==h,name
    raw=torch.load(plan['input'],weights_only=True,map_location='cpu')
    rows=[r for r in raw if not any(s['errors'] for s in r['samples'])]
    train=[r for r in rows if r['qid'] in plan['fit']];monitor=[r for r in rows if r['qid'] in plan['monitor']]
    assert not set(plan['fit'])&set(plan['monitor'])
    out=ROOT/NAME;out.mkdir(exist_ok=False);save(out/'plan.json',plan);start=time.monotonic()
    x,m,q,c,w=tensors(train);model=FutureCritic();initial_actor={k:v.clone() for k,v in model.actor.state_dict().items()}
    # Regression test: unavailable target contents cannot affect loss or gradients.
    testq=q.clone();testc=c.clone();testq[~m]=1e30;testc[~m]=-1e30
    pq,pc,_=model(x);a=masked_loss(pq,pc,q,c,m,w);b=masked_loss(pq,pc,testq,testc,m,w)
    assert torch.equal(a,b)
    a.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());model.zero_grad()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.01)
    curve=[]
    for step in range(100):
        opt.zero_grad();pq,pc,_=model(x);loss=masked_loss(pq,pc,q,c,m,w);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%20==0 or step==99:curve.append([step,float(loss.detach())])
    assert all(torch.equal(v,initial_actor[k]) for k,v in model.actor.state_dict().items())
    mx,mm,mq,mc,mw=tensors(monitor)
    with torch.no_grad():
        pq,pc,actor=model(mx);tq,tc,_=model(x)
    assert torch.count_nonzero(actor)==0
    branchw=w[:,None]*m/m.sum(-1,keepdim=True)
    baselineq=(torch.nan_to_num(q)*branchw).sum(0)/branchw.sum(0)
    baselinec=(torch.nan_to_num(c)*branchw).sum(0)/branchw.sum(0)
    actual=metrics(monitor,pq,pc);base=metrics(monitor,baselineq.expand_as(pq),baselinec.expand_as(pc))
    enough=actual['informative_pairs']>=10 and actual['informative_questions']>=5
    passed=enough and actual['gap_mse']<base['gap_mse'] and actual['cost_mae']<base['cost_mae'] and actual['gap_sign_accuracy']>.5
    torch.save(model.state_dict(),out/'critic.pt')
    reloaded=FutureCritic();reloaded.load_state_dict(torch.load(out/'critic.pt',weights_only=True))
    with torch.no_grad():assert torch.equal(reloaded(mx)[0],pq)
    predictions=[]
    for i,r in enumerate(monitor):
        predictions.append(dict(qid=r['qid'],turn=r['turn'],prefix_sha256=r['prefix_sha256'],branches=[dict(action=s['action'],f1=s['f1'],generated=s['generated'],predicted_f1=float(pq[i,s['action']]),predicted_generated=float(pc[i,s['action']]),baseline_f1=float(baselineq[s['action']]),baseline_generated=float(baselinec[s['action']])) for s in r['samples']]))
    save(out/'predictions.json',predictions)
    result=dict(status='complete',fit_questions=len(set(r['qid'] for r in train)),monitor_questions=len(set(r['qid'] for r in monitor)),fit_states=len(train),monitor_states=len(monitor),excluded=[dict(qid=r['qid'],turn=r['turn']) for r in raw if any(s['errors'] for s in r['samples'])],monitor=actual,baseline=base,train=metrics(train,tq,tc),gate_passed=passed,coverage_sufficient=enough,curve=curve,actor_unchanged=True,masked_loss_verified=True,checkpoint_sha256=sha(out/'critic.pt'),seconds=time.monotonic()-start,parameters=sum(p.numel() for p in model.parameters()),new_qa_episodes=0)
    save(out/'summary.json',result);save(out/'status.json',dict(phase='complete',pid=None));save(REPO/'lab/reports'/NAME/'summary.json',result);save(REPO/'lab/reports'/NAME/'predictions.json',predictions)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
