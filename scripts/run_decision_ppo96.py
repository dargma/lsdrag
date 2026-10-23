"""Bounded tempered PPO versus same-batch return-weighted imitation."""
import argparse,copy,json,hashlib,os,re,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from lsdrag.decision_policy import PolicyHead,PolicyReader,rollout,distribution,returns_for_episode,clipped_policy_loss
from lsdrag.text_decision_benchmark import public_paragraphs
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
from run_decision_policy_pilot import ROOT,REPO,PARENT,ORIGIN,sha,save
OLD=ROOT/'results/decision_policy_20261023_pilot2';REG=REPO/'lab/registry/decision_ppo96_20261023';OUT=ROOT/'results/decision_ppo96_20261023'
def read(p):return json.loads(p.read_text())
def prepare():
 old=read(OLD/'manifest.json');exposed=set()
 for folder in [REPO/'lab',ROOT/'results']:
  for p in folder.rglob('*.json'):exposed.update(re.findall(r'\b[234]hop__\d+_\d+\b',p.read_text()))
 data={name:[json.loads(x) for x in (ROOT/f'source/musique/musique_ans_v1.0_{name}.jsonl').read_text().splitlines()] for name in ['train','dev']}
 families={d['id'] for q in data['train']+data['dev'] if q['id'] in exposed for d in q['question_decomposition']};test=[];used=set()
 for q in sorted(data['dev'],key=lambda q:hashlib.sha256(('tempered-ppo:'+q['id']).encode()).hexdigest()):
  f={d['id'] for d in q['question_decomposition']}
  if q['answerable'] and len(f)==2 and q['id'] not in exposed and not f&(families|used):test.append(q['id']);used.update(f)
  if len(test)==40:break
 assert len(test)==40
 source=['scripts/run_decision_ppo96.py','src/lsdrag/decision_policy.py','src/lsdrag/text_integrated.py','src/lsdrag/text_integrated_aligned.py','prompts/text_rag_integrated_v5.txt']
 inputs=[OLD/'reference_round0.pt',ORIGIN/'head.pt']
 m={'seed':20261033,'train':old['train'],'test':test,'model':old['model'],'iterations':3,'repeats':2,'epochs':4,'lr':.0001,'clip':.2,'cost_weight':.03,'epsilon':0.,'anchor_KL':.02,'entropy_coef':.01,'rwr_temperature':.1,'batch_size':32,'policies':['historical_bc','tempered_anchor','ppo','rwr'],'trainable':'action, skip, value; trunk frozen; value zero initialized','budget_episodes':304,'max_seconds':1800,'max_generated':400000,'max_inputs':12000000,'comparison':'PPO on-policy collector supplies same batches to RWR; not independent on-policy RWR; collection-policy asymmetry declared','primary':'ppo vs tempered_anchor quality and utility; ppo vs rwr auxiliary','fresh_test_frozen':True,'single_seed':True,'no_validation_selection':True,'source_sha256':{p:sha(REPO/p) for p in source},'input_sha256':{str(p):sha(p) for p in inputs}}
 assert not (REG/'manifest.json').exists();save(REG/'manifest.json',m);print('Frozen 24 train / 40 fresh test / 304 episodes')
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
  reader=PolicyReader(m['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384));embed=V11ProcessEmbedder();deadline=time.perf_counter()+m['max_seconds'];prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text();cache={}
  anchor=PolicyHead(2560,'mlp').cuda();anchor.load_state_dict(torch.load(OLD/'reference_round0.pt',weights_only=True,map_location='cuda'));anchor.eval()
  heads={name:copy.deepcopy(anchor) for name in ['ppo','rwr']};opts={}
  for name,h in heads.items():
   with torch.no_grad():h.value.weight.zero_();h.value.bias.zero_()
   for n,p in h.named_parameters():p.requires_grad_(n.startswith(('action.','skip.','value.')))
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
     torch.manual_seed(m['seed']+iteration*10000+qi*10+rep);ep,steps=episode(qid,heads['ppo'],f'train_i{iteration}_r{rep}',True)
     for s,ret in zip(steps,returns_for_episode(steps,ep['f1'],.03)):
      if not s['forced']:batch.append(s|{'return':ret})
    phase('collection',iteration=iteration,questions=qi+1)
   x=torch.stack([s['pack'] for s in batch]).float().cuda();legal=torch.stack([s['legal'] for s in batch]).cuda();act=torch.tensor([s['action'] for s in batch],device='cuda');old=torch.tensor([s['old_logp'] for s in batch],device='cuda');ret=torch.tensor([s['return'] for s in batch],device='cuda');adv=ret-torch.tensor([s['value'] for s in batch],device='cuda');adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
   with torch.no_grad():
    before=distribution(heads['ppo'](x,legal)[0],legal).log_prob(act);assert torch.allclose(before,old,atol=2e-4,rtol=2e-4),'behavior logprob mismatch'
    ap=distribution(anchor(x,legal)[0],legal).probs;weights=((ret-ret.mean())/.1).clamp(-5,5).exp();weights/=weights.mean()
   torch.save(batch,OUT/f'batch_{iteration}.pt');diagnostic={}
   for name,h in heads.items():
    losses=[]
    for epoch in range(4):
     torch.manual_seed(m['seed']+iteration*100+epoch)
     for idx in torch.randperm(len(batch),device='cuda').split(32):
      opts[name].zero_grad();logits,v,_,_=h(x[idx],legal[idx]);dist=distribution(logits,legal[idx]);prob=dist.probs;kl=(prob*(prob.clamp_min(1e-9).log()-ap[idx].clamp_min(1e-9).log())).sum(-1).mean()
      actor=clipped_policy_loss(dist.log_prob(act[idx]),old[idx],adv[idx],.2) if name=='ppo' else -(weights[idx]*dist.log_prob(act[idx])).mean()
      loss=actor+.5*F.mse_loss(v,ret[idx])+.02*kl-.01*dist.entropy().mean();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in h.parameters() if p.requires_grad],1.);opts[name].step();losses.append(float(loss))
    with torch.no_grad():
     probs=distribution(h(x,legal)[0],legal).probs;changed=(probs.argmax(-1)!=ap.argmax(-1)).float().mean();ratio=(probs.gather(1,act[:,None]).squeeze().log()-old).exp()
    diagnostic[name]={'loss':float(np.mean(losses)),'changed_vs_anchor':float(changed),'ratio_clip_fraction':float(((ratio-1).abs()>.2).float().mean())}
    torch.save(h.state_dict(),OUT/f'{name}_iteration{iteration}.pt')
   curves.append({'iteration':iteration,'states':len(batch),'diagnostic':diagnostic});save(OUT/'curves.json',curves);phase('updated',iteration=iteration)
  evaluation=[]
  save(OUT/'frozen_evaluation.json',{'test':m['test'],'head_sha256':{name:sha(OUT/f'{name}_iteration2.pt') for name in heads}})
  for qi,qid in enumerate(m['test']):
   for name in m['policies']:
    h=anchor if name=='tempered_anchor' else heads.get(name);ep,_=episode(qid,h,f'test_{name}',historical=historical if name=='historical_bc' else None);ep['condition']=name;evaluation.append(ep)
   phase('evaluation',questions=qi+1)
  summary={}
  for name in m['policies']:
   es=[e for e in evaluation if e['condition']==name];summary[name]={k:float(np.mean([e[k] for e in es])) for k in ['f1','tool_calls','cost_units','generated_tokens','prompt_tokens']};summary[name].update(errors=sum(bool(e['errors']) for e in es),forced_final=sum(e['forced_final'] for e in es));summary[name]['utility']=summary[name]['f1']-.03*summary[name]['cost_units']
  comparisons={};rng=np.random.default_rng(m['seed']);draw=rng.integers(40,size=(4000,40))
  for name in ['historical_bc','tempered_anchor','rwr']:
   comparisons[name]={}
   for metric in ['f1','utility']:
    def value(e):return e['f1'] if metric=='f1' else e['f1']-.03*e['cost_units']
    delta=np.array([value(next(e for e in evaluation if e['qid']==q and e['condition']=='ppo'))-value(next(e for e in evaluation if e['qid']==q and e['condition']==name)) for q in m['test']]);comparisons[name][metric]={'delta':float(delta.mean()),'ci95':np.quantile(delta[draw].mean(1),[.025,.975]).tolist()}
  save(OUT/'summary.json',{'policies':summary,'ppo_minus':comparisons,'counts':counts,'single_seed':True,'controlled_latency':False});phase('complete')
 except BaseException as e:phase('failed',error=repr(e));raise
 finally:
  if embed:embed.close()
if __name__=='__main__':main()
