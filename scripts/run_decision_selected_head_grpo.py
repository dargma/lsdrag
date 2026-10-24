"""Continue GRPO from the already outcome-trained selected multi-position head.
Validation only; explicitly not token-budget-matched to branch collection.
"""
import argparse,copy,json,hashlib,os,re,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from lsdrag.decision_policy import PolicyHead,PolicyReader,rollout,distribution,returns_for_episode,clipped_policy_loss
from lsdrag.text_decision_benchmark import public_paragraphs
from lsdrag.decision_policy_recovery import RecoveryPolicyReader
from lsdrag.decision_grpo import group_advantages
from fit_decision_state_heads import load_head
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
from run_decision_policy_pilot import ROOT,REPO,PARENT,ORIGIN,sha,save
OLD=ROOT/'results/decision_policy_20261023_pilot2';REG=REPO/'lab/registry/decision_selected_head_grpo192_20261024';OUT=ROOT/'results/decision_selected_head_grpo192_20261024'
def read(p):return json.loads(p.read_text())
def continuation_heads(spec, device='cpu'):
 assert spec['kind']=='structure' and spec['inputs']=='multi' and spec['body']=='mlp'
 assert sha(Path(spec['checkpoint']))==spec['sha256']
 anchor=load_head(spec,device).requires_grad_(False)
 learner=copy.deepcopy(anchor).requires_grad_(True)
 assert all(n.startswith('net.') for n,_ in learner.named_parameters())
 return anchor,learner

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
 if a.prepare:raise RuntimeError('Use frozen PPO96 manifest; legacy prepare is disabled')
 m=read(REG/'manifest.json');assert all(sha(REPO/p)==h for p,h in m['source_sha256'].items());assert all(sha(Path(p))==h for p,h in m['input_sha256'].items())
 import fcntl
 lock=(ROOT/'decision_policy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(parents=True,exist_ok=False);save(OUT/'manifest.json',m)
 torch.set_num_threads(2);torch.manual_seed(m['seed']);start=time.perf_counter();embed=None;counts={'episodes':0,'generated':0,'inputs':0}
 def phase(name,**kw):save(OUT/'status.json',{'phase':name,'pid':None if name in ['complete','failed'] else os.getpid(),**counts,'seconds':time.perf_counter()-start,**kw})
 phase('loading');wanted=set(m['train']);raw={}
 for name in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{name}.jsonl').open():
   q=json.loads(line)
   if q['id'] in wanted:raw[q['id']]=q
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1
 try:
  reader=RecoveryPolicyReader(m['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384));embed=V11ProcessEmbedder();deadline=time.perf_counter()+m['max_seconds'];prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text();cache={}
  anchor,learner=continuation_heads(m['reference_spec'],'cuda')
  heads={'grpo':learner};opts={'grpo':torch.optim.AdamW(learner.parameters(),lr=m['lr'])}
  rows=[json.loads(line) for line in (Path(m['corpus'])/'documents.jsonl').open()];vec=np.load(Path(m['corpus'])/'vectors.npy',mmap_mode='r')
  def episode(qid,h,tag,stochastic=False,historical=None):
   assert time.perf_counter()<deadline and counts['episodes']<m['budget_episodes'] and counts['generated']+4608<=m['max_generated'] and counts['inputs']+148032<=m['max_inputs']
   q=raw[qid]
   ep,steps,_=rollout(reader,q['question'],rows,vec,embed,prompt,h,deadline,stochastic=stochastic,epsilon=0.,historical=historical)
   ep.update(qid=qid,policy=tag,f1=max(compute_f1(x,ep['answer']) for x in [q['answer']]+q['answer_aliases']));counts['episodes']+=1;counts['generated']+=ep['generated_tokens'];counts['inputs']+=ep['prompt_tokens'];save(OUT/f'episodes/{tag}_{qid}.json',ep);return ep,steps
  curves=[]
  for iteration in range(m['iterations']):
   batch=[]
   for qi,qid in enumerate(m['train']):
    for rep in range(m['repeats']):
     torch.manual_seed(m['seed']+iteration*10000+qi*10+rep);ep,steps=episode(qid,heads['grpo'],f'train_i{iteration}_r{rep}',True)
     for s,ret in zip(steps,returns_for_episode(steps,ep['f1'],.03)):
      if not s['forced']:batch.append(s|{'return':ret,'qid':qid,'rep':rep,'episode_reward':ep['f1']-.03*ep['cost_units'],'episode_steps':sum(not t['forced'] for t in steps)})
    phase('collection',iteration=iteration,questions=qi+1)
   x=torch.stack([s['pack'] for s in batch]).float().cuda();legal=torch.stack([s['legal'] for s in batch]).cuda();act=torch.tensor([s['action'] for s in batch],device='cuda');old=torch.tensor([s['old_logp'] for s in batch],device='cuda');ret=torch.tensor([s['return'] for s in batch],device='cuda');group={}
   for qid in m['train']:
    rewards=[next(s['episode_reward'] for s in batch if s['qid']==qid and s['rep']==rep) for rep in range(m['repeats'])];group[qid]=group_advantages(rewards)
   adv=torch.tensor([group[s['qid']][s['rep']] for s in batch],device='cuda',dtype=torch.float);roww=torch.tensor([len(batch)/(len(m['train'])*m['repeats']*s['episode_steps']) for s in batch],device='cuda')
   with torch.no_grad():
    before=distribution(heads['grpo'](x,legal)[0],legal).log_prob(act);assert torch.allclose(before,old,atol=2e-4,rtol=2e-4),'behavior logprob mismatch'
    ap=distribution(anchor(x,legal)[0],legal).probs
   torch.save(batch,OUT/f'batch_{iteration}.pt');diagnostic={}
   for name,h in heads.items():
    losses=[]
    for epoch in range(4):
     torch.manual_seed(m['seed']+iteration*100+epoch)
     for idx in torch.randperm(len(batch),device='cuda').split(32):
      opts[name].zero_grad();logits,v,_,_=h(x[idx],legal[idx]);dist=distribution(logits,legal[idx]);prob=dist.probs;kl=(prob*(prob.clamp_min(1e-9).log()-ap[idx].clamp_min(1e-9).log())).sum(-1)
      ratio=(dist.log_prob(act[idx])-old[idx]).exp();actor=-torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx])
      loss=(roww[idx]*(actor+.02*kl-.01*dist.entropy())).mean();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in h.parameters() if p.requires_grad],1.);opts[name].step();losses.append(float(loss.detach()))
    with torch.no_grad():
     probs=distribution(h(x,legal)[0],legal).probs;changed=(probs.argmax(-1)!=ap.argmax(-1)).float().mean();ratio=(probs.gather(1,act[:,None]).squeeze().log()-old).exp()
    diagnostic[name]={'loss':float(np.mean(losses)),'changed_vs_anchor':float(changed),'ratio_clip_fraction':float(((ratio-1).abs()>.2).float().mean())}
    torch.save(h.state_dict(),OUT/f'{name}_iteration{iteration}.pt')
   curves.append({'iteration':iteration,'states':len(batch),'diagnostic':diagnostic});save(OUT/'curves.json',curves);phase('updated',iteration=iteration)
  save(OUT/'summary.json',{'counts':counts,'iterations':m['iterations'],'curves':curves,'checkpoint_sha256':sha(OUT/f'grpo_iteration{m["iterations"]-1}.pt'),'validation_evaluated':False,'test_evaluated':False});phase('complete')
 except BaseException as e:phase('failed',error=repr(e));raise
 finally:
  if embed:embed.close()
if __name__=='__main__':main()
