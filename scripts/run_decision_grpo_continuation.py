"""Additional ordinary GRPO training control, initialized from frozen GRPO96.
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
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
from run_decision_policy_pilot import ROOT,REPO,PARENT,ORIGIN,sha,save
OLD=ROOT/'results/decision_policy_20261023_pilot2';REG=REPO/'lab/registry/decision_grpo_continuation_20261023';OUT=ROOT/'results/decision_grpo_continuation_20261023'
def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
 if a.prepare:raise RuntimeError('Use frozen PPO96 manifest; legacy prepare is disabled')
 m=read(REG/'manifest.json');assert all(sha(REPO/p)==h for p,h in m['source_sha256'].items());assert all(sha(Path(p))==h for p,h in m['input_sha256'].items())
 import fcntl
 lock=(ROOT/'decision_policy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(parents=True,exist_ok=False);save(OUT/'manifest.json',m)
 torch.set_num_threads(2);torch.manual_seed(m['seed']);start=time.perf_counter();embed=None;counts={'episodes':0,'generated':0,'inputs':0}
 def phase(name,**kw):save(OUT/'status.json',{'phase':name,'pid':None if name in ['complete','failed'] else os.getpid(),**counts,'seconds':time.perf_counter()-start,**kw})
 phase('loading');wanted=set(m['train']+m['test']);raw={}
 for name in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{name}.jsonl').open():
   q=json.loads(line)
   if q['id'] in wanted:raw[q['id']]=q
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1
 try:
  reader=RecoveryPolicyReader(m['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384));embed=V11ProcessEmbedder();deadline=time.perf_counter()+m['max_seconds'];prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text();cache={}
  anchor=PolicyHead(2560,'mlp').cuda();anchor.load_state_dict(torch.load(ROOT/'results/decision_grpo96_20261023/grpo_iteration2.pt',weights_only=True,map_location='cuda'));anchor.eval()
  heads={'grpo':copy.deepcopy(anchor)};opts={}
  for name,h in heads.items():
   with torch.no_grad():h.value.weight.zero_();h.value.bias.zero_()
   for n,p in h.named_parameters():p.requires_grad_(n.startswith(('action.','skip.')))
   opts[name]=torch.optim.AdamW([p for p in h.parameters() if p.requires_grad],lr=m['lr'])
  hist=torch.load(ORIGIN/'head.pt',weights_only=True,map_location='cuda');historical={k:hist[k] for k in ['action_weight','action_bias']}
  def episode(qid,h,tag,stochastic=False,historical=None):
   assert time.perf_counter()<deadline and counts['episodes']<m['budget_episodes'] and counts['generated']+4608<=m['max_generated'] and counts['inputs']+148032<=m['max_inputs']
   q=raw[qid]
   if qid not in cache:
    rows=public_paragraphs(q);p=PARENT/'index'/f'{qid}.npy';vec=np.load(p) if p.exists() else embed.encode([r['text'] for r in rows],'document',titles=[r['title'] for r in rows]);cache[qid]=(rows,vec)
   rows,vec=cache[qid];ep,steps,_=rollout(reader,q['question'],rows,vec,embed,prompt,h,deadline,stochastic=stochastic,epsilon=0.,historical=historical)
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
  evaluation=[]
  save(OUT/'frozen_evaluation.json',{'test':m['test'],'head_sha256':{'grpo':sha(OUT/'grpo_iteration2.pt')}})
  for qi,qid in enumerate(m['test']):
   for name in m['policies']:
    h=anchor if name=='tempered_anchor' else heads.get(name);ep,_=episode(qid,h,f'test_{name}',historical=historical if name=='historical_bc' else None);ep['condition']=name;evaluation.append(ep)
   phase('evaluation',questions=qi+1)
  summary={}
  for name in m['policies']:
   es=[e for e in evaluation if e['condition']==name];summary[name]={k:float(np.mean([e[k] for e in es])) for k in ['f1','tool_calls','cost_units','generated_tokens','prompt_tokens']};summary[name].update(errors=sum(bool(e['errors']) for e in es),forced_final=sum(e['forced_final'] for e in es));summary[name]['utility']=summary[name]['f1']-.03*summary[name]['cost_units']
  comparisons={};rng=np.random.default_rng(m['seed']);draw=rng.integers(len(m['test']),size=(4000,len(m['test'])))
  for name in ['tempered_anchor']:
   comparisons[name]={}
   for metric in ['f1','utility']:
    def value(e):return e['f1'] if metric=='f1' else e['f1']-.03*e['cost_units']
    delta=np.array([value(next(e for e in evaluation if e['qid']==q and e['condition']=='grpo'))-value(next(e for e in evaluation if e['qid']==q and e['condition']==name)) for q in m['test']]);comparisons[name][metric]={'delta':float(delta.mean()),'ci95':np.quantile(delta[draw].mean(1),[.025,.975]).tolist()}
  save(OUT/'summary.json',{'policies':summary,'grpo_minus':comparisons,'counts':counts,'single_seed':True,'controlled_latency':False});phase('complete')
 except BaseException as e:phase('failed',error=repr(e));raise
 finally:
  if embed:embed.close()
if __name__=='__main__':main()
