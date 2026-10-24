"""Two fixed, train-only 9B hypotheses; no development-based checkpoint selection."""
import copy,json,hashlib,os,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from fit_qwen35_outcome_heads import base_load,teacher_retention
from lsdrag.decision_policy_followup import state_weights
ROOT=Path('/content/local_fast/lsdrag-text-v1'); REPO=Path(__file__).resolve().parents[1]
NAME='qwen35_9b_two_promising_20261024'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 plan=read(REPO/'lab/registry'/NAME/'plan.json')
 for p,h in plan['hashes'].items():assert sha(p)==h,p
 torch.set_num_threads(2);torch.manual_seed(plan['seed'])
 out=ROOT/'results'/NAME;out.mkdir(exist_ok=False);start=time.monotonic()
 save(out/'status.json',dict(phase='training',pid=os.getpid()))
 try:
  records=torch.load(plan['states'],weights_only=True,map_location='cpu');base=base_load(plan['checkpoint'])
  assert {r['qid'] for r in records}==set(plan['train48'])
  assert not set(plan['train192'])&set(plan['dev64'])
  labels={}
  for i,r in enumerate(records):
   ss={s['action']:s for s in r['samples']};anchor=max(range(4),key=lambda a:r['reference_probs'][a]);ref=ss[anchor]
   choices=[a for a,s in ss.items() if not s['errors'] and not ref['errors'] and s['em']>=ref['em'] and s['f1']>=ref['f1'] and s['f1']>0 and (s['f1']>ref['f1'] or s['generated']<ref['generated'])]
   if choices:
    target=min(choices,key=lambda a:(-ss[a]['f1'],ss[a]['generated'],a))
    if target!=anchor:labels[i]=target
  assert len({records[i]['qid'] for i in labels})>=3
  seen={(r['qid'],r['prefix_sha256']) for r in records}
  retain=[r for r in teacher_retention() if (r['qid'],r['prefix_sha256']) not in seen]+[r for i,r in enumerate(records) if i not in labels]
  def tensors(rows):return torch.stack([r['pack'] for r in rows]).float(),torch.stack([r['legal'] for r in rows]).bool(),state_weights([r['qid'] for r in rows],'cpu')
  cx,cm,cw=tensors([records[i] for i in labels]);rx,rm,rw=tensors(retain);x,m,w=tensors(records)
  q=torch.zeros(len(records),4);cost=q.clone();valid=m.clone()
  for i,r in enumerate(records):
   for s in r['samples']:
    a=s['action'];q[i,a]=s['f1'];cost[i,a]=s['generated'];valid[i,a]=not bool(s['errors'])
  with torch.no_grad():
   target=.5*base(cx,cm)[0].softmax(-1)+.5*F.one_hot(torch.tensor(list(labels.values())),4).float();rt=base(rx,rm)[0].softmax(-1);before=base(x,m)[0].argmax(-1)
  models={}
  for name in ['h1_conservative','value_aux']:
   torch.manual_seed(plan['seed']);h=copy.deepcopy(base)
   if name=='h1_conservative':
    for p in h.net[0].parameters():p.requires_grad_(False)
   aux=nn.Linear(128,8) if name=='value_aux' else None
   params=[p for p in h.parameters() if p.requires_grad]+([] if aux is None else list(aux.parameters()))
   opt=torch.optim.AdamW(params,lr=plan['lr'])
   for epoch in range(plan['epochs']):
    lc=-(target*h(cx,cm)[0].log_softmax(-1)).sum(-1);lr=-(rt*h(rx,rm)[0].log_softmax(-1)).sum(-1)
    policy=.5*(cw*lc).sum()+.5*(rw*lr).sum();aloss=policy.new_zeros(())
    if aux is not None:
     z=h.net[1](h.net[0]((h.features(x)-h.mean)/h.scale));pred=aux(z)
     sq=(pred[:,:4].sigmoid()-q).square();sc=F.smooth_l1_loss(F.softplus(pred[:,4:]),torch.log1p(cost),reduction='none')
     aloss=(w*((sq+sc).masked_fill(~valid,0).sum(-1)/valid.sum(-1).clamp_min(1))).sum()
    loss=policy+plan['aux_coef']*aloss;opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
   assert torch.equal(h.mean,base.mean) and torch.equal(h.scale,base.scale)
   if name=='h1_conservative':assert all(torch.equal(v,h.net[0].state_dict()[k]) for k,v in base.net[0].state_dict().items())
   assert all(torch.isfinite(v).all() for v in h.state_dict().values())
   path=out/f'{name}.pt';torch.save(h.state_dict(),path)
   if aux is not None:torch.save(aux.state_dict(),out/'value_aux_training_predictor.pt')
   with torch.no_grad():changed=int((h(x,m)[0].argmax(-1)!=before).sum())
   models[name]=dict(kind='qwen35_base',checkpoint=str(path),sha256=sha(path),trainable_parameters=sum(p.numel() for p in params),policy_loss=float(policy.detach()),aux_loss=float(aloss.detach()),changed_train_states=changed)
  manifest=dict(models=models,corrections=len(labels),retention=len(retain),states=len(records),plan=plan,aux_inference=False,note='Both use H1 targets. Value candidate jointly learns representation and auxiliary outcomes: bundled hypothesis, not isolated auxiliary ablation.')
  save(out/'manifest.json',manifest);save(out/'plan.json',plan);save(out/'status.json',dict(phase='complete',pid=None,seconds=time.monotonic()-start))
  save(REPO/'lab/reports'/NAME/'training.json',manifest)
  print(json.dumps(models,indent=2))
 except BaseException as e:save(out/'status.json',dict(phase='failed',pid=None,error=repr(e)));raise
if __name__=='__main__':main()
