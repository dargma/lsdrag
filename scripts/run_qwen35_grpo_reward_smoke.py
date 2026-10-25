"""One bounded GRPO smoke: correctness, total emitted length, schema validity."""
import copy,json,os,time,sys,hashlib,fcntl
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from jsonschema import validate
from lsdrag.decision_policy import rollout,distribution
from lsdrag.decision_grpo import group_advantages
from lsdrag.decision_qwen35_reader import Qwen35Reader
from lsdrag.text_integrated import argument_schema,ACTIONS
from lsdrag.text_decision_benchmark import FINAL_SCHEMA
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
from fit_qwen35_outcome_heads import base_load
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[1];NAME='qwen35_grpo_reward_smoke_20261025'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def reward(em,f1,tokens,valid):return float(em+.25*f1-.10*tokens/(tokens+256)+.02*valid)
def format_valid(ep):
 try:
  calls=[u for u in ep['reader_calls'] if 'raw_json' in u];ds=ep['decisions']
  if ep['errors'] or len(calls)!=len(ds) or not ds or ds[-1]['action']!=3:return False
  history=[]
  for u,d in zip(calls,ds):
   name=ACTIONS[d['action']];obj=json.loads(u['raw_json']);validate(obj,FINAL_SCHEMA if name=='FINAL' else argument_schema(name,history))
   if u['finish_reason']!='schema_complete':return False
   if name!='FINAL':history.append(ep['trajectory'][len(history)])
  return obj=={'answer':ep['answer'],'support_ids':ep['support_ids']}
 except (ValueError,KeyError,IndexError,TypeError):return False
 except Exception as e:
  if type(e).__name__=='ValidationError':return False
  raise

def main():
 p=read(REPO/'lab/registry'/NAME/'plan.json')
 for f,h in p['hashes'].items():assert sha(f)==h,f
 assert reward(1,1,100,1)>reward(1,1,200,1)
 assert reward(1,1,1000000,0)>reward(0,1,0,1)
 lock=(ROOT/'decision_policy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=ROOT/'results'/NAME;out.mkdir(exist_ok=False);save(out/'plan.json',p);start=time.monotonic();counts={'episodes':0,'generated':0,'inputs':0};embed=None
 def status(phase,**kw):save(out/'status.json',dict(phase=phase,pid=None if phase in ['complete','failed'] else os.getpid(),seconds=time.monotonic()-start,**counts,**kw))
 status('loading');torch.set_num_threads(2);torch.manual_seed(p['seed'])
 try:
  raw={q['id']:q for q in map(json.loads,(ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open()) if q['id'] in p['qids']};assert len(raw)==2
  sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1,compute_exact
  anchor=base_load(p['checkpoint']).cuda().requires_grad_(False);learner=copy.deepcopy(anchor).requires_grad_(True)
  reader=Qwen35Reader(p['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384));assert not any(x.requires_grad for x in reader.model.parameters())
  embed=V11ProcessEmbedder();rows=list(map(json.loads,(Path(p['corpus'])/'documents.jsonl').open()));vec=np.load(Path(p['corpus'])/'vectors.npy',mmap_mode='r');prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text();deadline=time.perf_counter()+p['max_seconds']
  def collect(qid,tag,head,stochastic=False):
   assert counts['episodes']<12 and counts['generated']+4608<=p['max_generated']
   ep,steps,_=rollout(reader,raw[qid]['question'],rows,vec,embed,prompt,head,deadline,stochastic=stochastic,epsilon=0.,state_storage_dtype=torch.float32)
   answers=[raw[qid]['answer']]+raw[qid]['answer_aliases'];em=max(compute_exact(a,ep['answer']) for a in answers);f1=max(compute_f1(a,ep['answer']) for a in answers);valid=format_valid(ep)
   ep.update(qid=qid,policy=tag,em=em,f1=f1,format_valid=valid,reward=reward(em,f1,ep['generated_tokens'],valid));save(out/'episodes'/f'{tag}_{qid}.json',ep)
   counts['episodes']+=1;counts['generated']+=ep['generated_tokens'];counts['inputs']+=ep['prompt_tokens'];status('collection',last=tag)
   return ep,steps
  before={q:collect(q,'before',anchor)[0] for q in p['qids']};batch=[];groups={};train=[]
  for qi,q in enumerate(p['qids']):
   eps=[]
   for rep in range(4):
    torch.manual_seed(p['seed']+qi*10+rep);ep,steps=collect(q,f'train{rep}',learner,True);eps.append(ep);train.append(ep);count=sum(not s['forced'] for s in steps);assert count>0
    batch += [s|{'qid':q,'rep':rep,'reward':ep['reward'],'episode_steps':count} for s in steps if not s['forced']]
   rewards=[e['reward'] for e in eps];groups[q]=dict(rewards=rewards,advantages=group_advantages(rewards).tolist(),format_values=[e['format_valid'] for e in eps])
  torch.save(batch,out/'batch.pt');save(out/'groups.json',groups)
  x=torch.stack([s['pack'] for s in batch]).float().cuda();legal=torch.stack([s['legal'] for s in batch]).cuda();a=torch.tensor([s['action'] for s in batch],device='cuda');old=torch.tensor([s['old_logp'] for s in batch],device='cuda');adv=torch.tensor([groups[s['qid']]['advantages'][s['rep']] for s in batch],device='cuda');w=torch.tensor([len(batch)/(8*s['episode_steps']) for s in batch],device='cuda')
  with torch.no_grad():
   dist=distribution(learner(x,legal)[0],legal);err=float((dist.log_prob(a)-old).abs().max());assert torch.allclose(dist.log_prob(a),old,atol=2e-4,rtol=2e-4);ref=distribution(anchor(x,legal)[0],legal).probs
  opt=torch.optim.AdamW(learner.parameters(),lr=p['lr']);losses=[]
  for epoch in range(2):
   torch.manual_seed(p['seed']+100+epoch)
   for idx in torch.randperm(len(batch),device='cuda').split(16):
    dist=distribution(learner(x[idx],legal[idx])[0],legal[idx]);prob=dist.probs;ratio=(dist.log_prob(a[idx])-old[idx]).exp();actor=-torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx]);kl=(prob*(prob.clamp_min(1e-9).log()-ref[idx].clamp_min(1e-9).log())).sum(-1);loss=(w[idx]*(actor+.02*kl)).mean();assert torch.isfinite(loss)
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(learner.parameters(),1.);opt.step();losses.append(float(loss.detach()))
  changed=[k for k,v in learner.state_dict().items() if not torch.equal(v,anchor.state_dict()[k])];assert changed and all(k.startswith('net.') for k in changed);assert all(torch.isfinite(v).all() for v in learner.state_dict().values())
  torch.save(learner.state_dict(),out/'head.pt');after={q:collect(q,'after',learner)[0] for q in p['qids']}
  summary=dict(smoke_passed=True,quality_evaluated=False,states=len(batch),optimizer_steps=len(losses),replay_max_error=err,changed_keys=changed,checkpoint_sha256=sha(out/'head.pt'),format_valid_train=sum(e['format_valid'] for e in train),format_total_train=len(train),format_group_variation=sum(len(set(g['format_values']))>1 for g in groups.values()),reward_group_variation=sum(max(g['rewards'])>min(g['rewards']) for g in groups.values()),counts=counts,before_after={q:{k:{f:ep[f] for f in ['answer','em','f1','generated_tokens','format_valid','reward']} for k,ep in [('before',before[q]),('after',after[q])]} for q in p['qids']},note='Train-only functional smoke; not dev quality. Format bonus cancels from GRPO advantages if all group outputs valid. Frozen reader/token generator; only action policy trained. No automatic full run.')
  save(out/'summary.json',summary);status('complete');print(json.dumps(summary,indent=2))
 except BaseException as e:status('failed',error=repr(e));raise
 finally:
  if embed:embed.close()
if __name__=='__main__':main()
