"""Matched one-batch policy update: historical reproduction, total and suffix tokens."""
import collections,json,time,os,fcntl
from pathlib import Path
import numpy as np
import torch
from run_decision_quality_head_grpo import continuation_heads,quality_gated_advantages
from lsdrag.decision_policy import distribution
from lsdrag.decision_step_credit import token_returns,group_step_advantages
from run_decision_policy_pilot import ROOT,REPO,save,sha
NAME='decision_reward_replay_20261024'
def main():
 p=json.loads((REPO/'lab/registry'/NAME/'plan.json').read_text())
 for f,h in p['source_sha256'].items():assert sha(REPO/f)==h,f
 for f,h in p['input_sha256'].items():assert sha(Path(f))==h,f
 for f,h in p['episode_sha256'].items():assert sha(Path(p['episodes'])/f)==h,f
 lock=(ROOT/'decision_policy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=ROOT/'results'/NAME;out.mkdir(exist_ok=False);save(out/'plan.json',p);start=time.monotonic()
 def phase(name,**kw):save(out/'status.json',dict(phase=name,pid=None if name in ['complete','failed'] else os.getpid(),seconds=time.monotonic()-start,new_llm_rollouts=0,**kw))
 phase('loading');torch.set_num_threads(2)
 try:
  m=json.loads(Path(p['original_manifest']).read_text());batch=torch.load(p['batch'],map_location='cpu',weights_only=True)
  assert {s['qid'] for s in batch}==set(m['train']);assert all(not s['forced'] and not s['branch_intervention'] for s in batch)
  groups={q:[json.loads((Path(p['episodes'])/f'train_i0_r{r}_{q}.json').read_text()) for r in range(4)] for q in m['train']}
  scale=float(np.std([token_returns(e,remaining=False)[0] for es in groups.values() for e in es]));assert scale>0
  x=torch.stack([s['pack'] for s in batch]).float().cuda();legal=torch.stack([s['legal'] for s in batch]).cuda();act=torch.tensor([s['action'] for s in batch],device='cuda');old=torch.tensor([s['old_logp'] for s in batch],device='cuda')
  w=torch.tensor([len(batch)/(len(m['train'])*4*s['episode_steps']) for s in batch],device='cuda')
  saved_adv=json.loads(Path(p['original_advantages']).read_text());candidates={};audits={}
  for mode in ['historical','token_total','token_remaining']:
   phase('training',candidate=mode);torch.manual_seed(m['seed']);anchor,h=continuation_heads(m['reference_spec'],'cuda');opt=torch.optim.AdamW(h.parameters(),lr=m['lr'])
   with torch.no_grad():
    dist=distribution(h(x,legal)[0],legal);error=float((dist.log_prob(act)-old).abs().max());assert torch.allclose(dist.log_prob(act),old,atol=2e-4,rtol=2e-4)
    ap=distribution(anchor(x,legal)[0],legal).probs
   if mode=='historical':ad=[saved_adv[s['qid']][s['rep']] for s in batch]
   else:
    values={q:group_step_advantages(es,scale,remaining=mode=='token_remaining') for q,es in groups.items()};counter=collections.Counter();ad=[]
    for s in batch:
     key=(s['qid'],s['rep']);idx=counter[key];ad.append(values[key[0]][key[1]][idx]);counter[key]+=1
    assert all(counter[q,r]==len(values[q][r]) for q in groups for r in range(4))
   adv=torch.tensor(ad,device='cuda',dtype=torch.float);losses=[]
   for epoch in range(4):
    torch.manual_seed(m['seed']+epoch)
    for idx in torch.randperm(len(batch),device='cuda').split(32):
     opt.zero_grad();logits=h(x[idx],legal[idx])[0];d=distribution(logits,legal[idx]);prob=d.probs;kl=(prob*(prob.clamp_min(1e-9).log()-ap[idx].clamp_min(1e-9).log())).sum(-1)
     ratio=(d.log_prob(act[idx])-old[idx]).exp();actor=-torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx]);loss=(w[idx]*(actor+.02*kl-.01*d.entropy())).mean();loss.backward();torch.nn.utils.clip_grad_norm_([v for v in h.parameters() if v.requires_grad],1.);opt.step();losses.append(float(loss.detach()))
   state={k:v.cpu() for k,v in h.state_dict().items()};ref={k:v.cpu() for k,v in anchor.state_dict().items()};changed=[k for k in state if not torch.equal(state[k],ref[k])]
   assert set(changed)=={'encoder.weight','encoder.bias','policy_weight','policy_bias'};assert all(torch.isfinite(v).all() for v in state.values())
   audit=dict(replay_error=error,changed=changed,updates=len(losses))
   if mode=='historical':
    expected=torch.load(p['original_checkpoint'],map_location='cpu',weights_only=True);delta=max(float((state[k]-expected[k]).abs().max()) for k in state);audit['max_checkpoint_error']=delta;audit['exact_checkpoint_match']=all(torch.equal(state[k],expected[k]) for k in state);save(out/'reproduction.json',audit)
    assert delta<=1e-6,'Historical reproduction mismatch; stop before candidate fitting'
   cp=out/f'{mode}.pt';torch.save(state,cp);candidates[mode]=dict(kind='decision_value',body='mlp',checkpoint=str(cp),sha256=sha(cp));audits[mode]=audit
   del h,anchor,opt;torch.cuda.empty_cache()
  save(out/'manifest.json',dict(candidates=candidates,audits=audits,scale=scale,plan=p,new_llm_rollouts=0));phase('complete')
 except BaseException as e:phase('failed',error=repr(e));raise
if __name__=='__main__':main()
