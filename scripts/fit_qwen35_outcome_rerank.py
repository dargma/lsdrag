"""One fixed outcome readout fit, question-separated predictive feasibility gate."""
import json,hashlib,time
from pathlib import Path
from collections import Counter,defaultdict
import torch
from torch.nn import functional as F
from lsdrag.decision_outcome_rerank import OutcomeRerankHead
from fit_qwen35_outcome_heads import base_load
from lsdrag.decision_policy_followup import state_weights
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[1];NAME='qwen35_outcome_rerank_20261025'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 plan=read(REPO/'lab/registry'/NAME/'plan.json');source=ROOT/'results/qwen35_final_grpo48_20261025';assert read(source/'status.json')['phase']=='complete';assert read(REPO/'lab/reports'/source.name/'audit.json')['verified']
 for p,h in plan['source_sha256'].items():assert sha(REPO/p)==h,p
 assert sha(plan['checkpoint'])==plan['checkpoint_sha256'];torch.set_num_threads(2);torch.manual_seed(plan['seed']);start=time.monotonic();out=ROOT/'results'/NAME;out.mkdir(exist_ok=False)
 records=torch.load(source/'batch.pt',weights_only=True,map_location='cpu');episodes={};offset=defaultdict(int);labels=[];counts=Counter();train_qs=set(plan['train32']);cal_qs=set(plan['calibration16']);assert not train_qs&cal_qs
 for r in records:
  q,rep=r['qid'],r['rep'];key=(q,rep)
  if key not in episodes:
   ep=read(source/'episodes'/f'train{rep}_{q}.json');usages=[u for u in ep['reader_calls'] if 'token_ids' in u];assert len(usages)==len(ep['decisions']);episodes[key]=(ep,usages)
  ep,usages=episodes[key];turn=offset[key];offset[key]+=1;assert not ep['decisions'][turn]['forced'];assert r['action']==ep['decisions'][turn]['action'];remaining=sum(u['generated_tokens'] for u in usages[turn:]);assert remaining>0
  labels.append(dict(qid=q,action=r['action'],em=ep['em'],remaining=remaining,turn=turn,rep=rep));counts[q]+=1
 assert len(episodes)==192
 def tensor(indices):
  rs=[records[i] for i in indices];ys=[labels[i] for i in indices]
  return torch.stack([r['pack'] for r in rs]).float(),torch.tensor([r['action'] for r in rs]),torch.tensor([y['em'] for y in ys],dtype=torch.float),torch.tensor([y['remaining'] for y in ys],dtype=torch.float).log1p(),state_weights([r['qid'] for r in rs],'cpu'),torch.stack([r['legal'] for r in rs])
 ti=[i for i,r in enumerate(records) if r['qid'] in train_qs];vi=[i for i,r in enumerate(records) if r['qid'] in cal_qs];assert len(ti)+len(vi)==len(records)
 x,a,y,c,w,m=tensor(ti);vx,va,vy,vc,vw,vm=tensor(vi);h=OutcomeRerankHead();h.base.load_state_dict(base_load(plan['checkpoint']).state_dict());h.base.requires_grad_(False)
 qbase=torch.zeros(4);cbase=torch.zeros(4);coverage={};valid_coverage={}
 with torch.no_grad():
  for act in range(4):
   mask=a==act;assert mask.any(),f'No train action{act}';qbase[act]=((w[mask]*y[mask]).sum()/w[mask].sum()).clamp(.01,.99);order=c[mask].argsort();cw=w[mask][order];ci=int(torch.searchsorted(cw.cumsum(0),cw.sum()/2));cbase[act]=c[mask][order][ci];coverage[act]=len({records[i]['qid'] for i in ti if records[i]['action']==act});valid_coverage[act]=len({records[i]['qid'] for i in vi if records[i]['action']==act})
  h.success.weight.zero_();h.success.bias.copy_(qbase.logit());h.logcost.weight.zero_();h.logcost.bias.copy_(torch.log(torch.expm1(cbase)));h.supported.copy_(torch.tensor([coverage[i]>=5 for i in range(4)]))
 opt=torch.optim.AdamW(list(h.success.parameters())+list(h.logcost.parameters()),lr=.001)
 for _ in range(100):
  pq,pc=h.predict(x);sq=pq.gather(1,a[:,None]).squeeze(1);sc=pc.gather(1,a[:,None]).squeeze(1);loss=(w*(F.binary_cross_entropy(sq,y,reduction='none')+F.smooth_l1_loss(sc,c,reduction='none'))).sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(h.success.parameters())+list(h.logcost.parameters()),1.);opt.step()
 with torch.no_grad():
  q,cost=h.predict(vx);sq=q.gather(1,va[:,None]).squeeze(1);sc=cost.gather(1,va[:,None]).squeeze(1)
  brier=float((vw*(sq-vy).square()).sum());baseline_brier=float((vw*(qbase[va]-vy).square()).sum());mae=float((vw*(sc-vc).abs()).sum());baseline_mae=float((vw*(cbase[va]-vc).abs()).sum())
  changed=int((h(vx,vm)[0].argmax(-1)!=h.base(vx,vm)[0].argmax(-1)).sum())
 gate=all(coverage[a]>=5 and valid_coverage[a]>=2 for a in range(4)) and brier<baseline_brier and mae<baseline_mae
 original=base_load(plan['checkpoint']);assert all(torch.equal(t,h.base.state_dict()[k]) for k,t in original.state_dict().items());assert all(torch.isfinite(t).all() for t in h.state_dict().values())
 cp=out/'head.pt';torch.save(h.state_dict(),cp);save(out/'labels.json',labels)
 summary=dict(fit_complete=True,predictive_gate_passed=gate,checkpoint=str(cp),checkpoint_sha256=sha(cp),source_batch_sha256=sha(source/'batch.pt'),training_states=len(ti),calibration_states=len(vi),coverage=coverage,calibration_coverage=valid_coverage,calibration_brier=brier,constant_action_brier=baseline_brier,calibration_logcost_mae=mae,constant_action_logcost_mae=baseline_mae,calibration_policy_changes=changed,base_tensors_identical=True,trainable_parameters=1032,quality_evaluated=False,scope='Question-separated new outcome labels; base representation previously saw these questions. Single-sampled action labels, no unseen-action causal certainty.')
 save(out/'summary.json',summary);save(out/'plan.json',plan);save(out/'status.json',dict(phase='complete',pid=None,seconds=time.monotonic()-start,predictive_gate_passed=gate));save(REPO/'lab/reports'/NAME/'fit.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
