"""Distill local empirical corrections while retaining anchor behavior elsewhere."""
import copy,json,os,time
from pathlib import Path
import torch
from lsdrag.decision_attention_adapter import AttentionAdapter
from lsdrag.decision_policy_followup import state_weights
from evaluate_decision_attention import load_head
from run_decision_policy_pilot import ROOT,REPO,save,sha
NAME='decision_cost_priority_20261024'
def read(p):return json.loads(p.read_text())
def main():
 plan=read(REPO/'lab/registry'/NAME/'plan.json')
 for f,d in plan['input_sha256'].items():assert sha(Path(f))==d,f
 for f,d in plan['source_sha256'].items():assert sha(REPO/f)==d,f
 out=ROOT/'results'/NAME;out.mkdir(exist_ok=False);save(out/'plan.json',plan);start=time.monotonic();torch.set_num_threads(2)
 def phase(name,**kw):save(out/'status.json',dict(phase=name,pid=None if name in ('complete','failed') else os.getpid(),seconds=time.monotonic()-start,**kw))
 phase('loading')
 try:
  fresh=torch.load(plan['fresh'],weights_only=True,map_location='cpu');old=torch.load(plan['old'],weights_only=True,map_location='cpu')
  previous=torch.load(plan['previous_features'],weights_only=True,map_location='cpu')
  old=list({(r['qid'],r['prefix_sha256']):r for r in old+previous}.values())
  assert not {r['qid'] for r in fresh+old}&set(plan['validation']+plan['test'])
  assert {r['qid'] for r in old}==set(plan['train_pool'])
  base=load_head(plan['reference_spec']);fx=torch.stack([r['pack'] for r in fresh]).float();fm=torch.stack([r['legal'] for r in fresh])
  with torch.no_grad():fp=base(fx,fm)[0].softmax(-1);actions=fp.argmax(-1).tolist()
  corrections={};cost_states=0
  for i,(r,b) in enumerate(zip(fresh,actions)):
   by={a:[s for s in r['samples'] if s['action']==a] for a in range(4)};bs=by[b];choices=[];cheaper=[]
   for a,ss in by.items():
    if a==b or len(ss)!=2 or len(bs)!=2 or any(s['errors'] for s in ss+bs):continue
    gap=min(s['f1'] for s in ss)-max(s['f1'] for s in bs)
    cost=max(s['generated'] for s in ss)-min(s['generated'] for s in bs)
    if gap>0 or (gap==0 and min(s['f1'] for s in ss)>0 and cost<0):choices.append((gap,-cost,-a,a))
    if gap>=0 and min(s['f1'] for s in ss)>0 and cost<0:cheaper.append((-cost,gap,-a,a))
   if cheaper:corrections[i]=max(cheaper)[3];cost_states+=1
   elif choices:corrections[i]=max(choices)[3]
  assert len(corrections)==21 and cost_states==14,'Unexpected candidate evidence'
  seen={(r['qid'],r['prefix_sha256']) for r in fresh}
  # Avoid contradictory anchor targets on the same prefix as a corrected state.
  retain=[r for r in old if (r['qid'],r['prefix_sha256']) not in seen]+[r for i,r in enumerate(fresh) if i not in corrections]
  rx=torch.stack([r['pack'] for r in retain]).float();rm=torch.stack([r['legal'] for r in retain]);rw=state_weights([r['qid'] for r in retain],'cpu')
  ids=list(corrections);cx=fx[ids];cm=fm[ids];cw=state_weights([fresh[i]['qid'] for i in ids],'cpu')
  with torch.no_grad():rp=base(rx,rm)[0].softmax(-1);target=.5*fp[ids];target[torch.arange(len(ids)),torch.tensor(list(corrections.values()))]+=.5
  save(out/'targets.json',dict(corrections=[dict(index=i,qid=fresh[i]['qid'],turn=fresh[i]['turn'],from_action=actions[i],to_action=a) for i,a in corrections.items()],retention_states=len(retain),cost_states=cost_states,mix=.5))
  candidates={};audits={}
  for body in ['mlp']:
   phase('training',candidate=body);torch.manual_seed(plan['seed']);head=copy.deepcopy(base) if body=='mlp' else AttentionAdapter(base,body)
   if body!='mlp':
    for p in head.base.parameters():p.requires_grad_(False)
   parameters=[p for p in head.parameters() if p.requires_grad];buffers={k:v.clone() for k,v in head.named_buffers()}
   with torch.no_grad():assert torch.equal(head(fx,fm)[0],base(fx,fm)[0])
   opt=torch.optim.AdamW(parameters,lr=1e-4);curve=[]
   for epoch in range(100):
    opt.zero_grad();lc=-(cw*(target*head(cx,cm)[0].log_softmax(-1)).sum(-1)).sum();lr=-(rw*(rp*head(rx,rm)[0].log_softmax(-1)).sum(-1)).sum();loss=.5*lc+.5*lr
    loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1);opt.step()
    if epoch%20==0 or epoch==99:curve.append(dict(epoch=epoch,correction_ce=float(lc.detach()),retention_ce=float(lr.detach())))
   assert all(torch.equal(v,buffers[k]) for k,v in head.named_buffers());assert all(torch.isfinite(v).all() for v in head.state_dict().values())
   if body!='mlp':assert all(torch.equal(v,base.state_dict()[k]) for k,v in head.base.state_dict().items())
   cp=out/(body+'.pt');torch.save(head.state_dict(),cp)
   desc=dict(kind='structure',inputs='multi',body='mlp') if body=='mlp' else dict(kind='attention_adapter',variant=body)
   candidates[body]=dict(**desc,checkpoint=str(cp),sha256=sha(cp));audits[body]=dict(curve=curve,initial_exact=True,normalization_fixed=True)
  save(out/'manifest.json',dict(candidates=candidates,audits=audits,plan=plan,new_llm_rollouts=0));phase('complete')
 except BaseException as e:phase('failed',error=repr(e));raise
if __name__=='__main__':main()
