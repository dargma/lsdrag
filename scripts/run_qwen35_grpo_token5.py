"""One frozen final GRPO48 round: correctness, total emitted length, schema validity."""
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
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[1];NAME='qwen35_grpo_token5_20261025'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
from run_qwen35_grpo_reward_smoke import format_valid
def reward(em,f1,t,valid):return em+.25*f1-.50*t/(t+256)+.02*valid

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
  raw={q['id']:q for q in map(json.loads,(ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open()) if q['id'] in p['qids']};assert len(raw)==48
  sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1,compute_exact
  anchor=base_load(p['checkpoint']).cuda().requires_grad_(False);learner=copy.deepcopy(anchor).requires_grad_(True)
  source=Path(p['reuse_run']);audit=read(REPO/'lab/reports/qwen35_final_grpo48_20261025/audit.json')
  assert audit['verified']
  train=[];groups={};lookup={}
  for path,h in audit['raw_sha256'].items():
   assert sha(path)==h
   e=read(path);e['reward']=reward(e['em'],e['f1'],e['generated_tokens'],e['format_valid'])
   save(out/'episodes'/Path(path).name,e);train.append(e);lookup[e['qid'],e['policy']]=e
  assert len(train)==192
  for q in p['qids']:
   eps=[lookup[q,f'train{i}'] for i in range(4)];rewards=[e['reward'] for e in eps]
   groups[q]=dict(rewards=rewards,advantages=group_advantages(rewards).tolist(),format_values=[e['format_valid'] for e in eps])
  batch=torch.load(source/'batch.pt',weights_only=True,map_location='cpu')
  for s in batch:s['reward']=lookup[s['qid'],f"train{s['rep']}"]['reward']
  counts.update(episodes=192,generated=sum(e['generated_tokens'] for e in train),inputs=sum(e['prompt_tokens'] for e in train))
  torch.save(batch,out/'batch.pt');save(out/'groups.json',groups)
  x=torch.stack([s['pack'] for s in batch]).float().cuda();legal=torch.stack([s['legal'] for s in batch]).cuda();a=torch.tensor([s['action'] for s in batch],device='cuda');old=torch.tensor([s['old_logp'] for s in batch],device='cuda');adv=torch.tensor([groups[s['qid']]['advantages'][s['rep']] for s in batch],device='cuda');w=torch.tensor([len(batch)/(192*s['episode_steps']) for s in batch],device='cuda')
  with torch.no_grad():
   dist=distribution(learner(x,legal)[0],legal);err=float((dist.log_prob(a)-old).abs().max());assert torch.allclose(dist.log_prob(a),old,atol=2e-4,rtol=2e-4);ref=distribution(anchor(x,legal)[0],legal).probs
  opt=torch.optim.AdamW(learner.parameters(),lr=p['lr']);losses=[]
  for epoch in range(2):
   torch.manual_seed(p['seed']+100+epoch)
   for idx in torch.randperm(len(batch),device='cuda').split(16):
    dist=distribution(learner(x[idx],legal[idx])[0],legal[idx]);prob=dist.probs;ratio=(dist.log_prob(a[idx])-old[idx]).exp();actor=-torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx]);kl=(prob*(prob.clamp_min(1e-9).log()-ref[idx].clamp_min(1e-9).log())).sum(-1);loss=(w[idx]*(actor+.02*kl)).mean();assert torch.isfinite(loss)
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(learner.parameters(),1.);opt.step();losses.append(float(loss.detach()))
  changed=[k for k,v in learner.state_dict().items() if not torch.equal(v,anchor.state_dict()[k])];assert changed and all(k.startswith('net.') for k in changed);assert all(torch.isfinite(v).all() for v in learner.state_dict().values())
  torch.save(learner.state_dict(),out/'head.pt')
  summary=dict(training_complete=True,quality_evaluated=False,states=len(batch),optimizer_steps=len(losses),replay_max_error=err,changed_keys=changed,checkpoint_sha256=sha(out/'head.pt'),format_valid_train=sum(e['format_valid'] for e in train),format_total_train=len(train),format_group_variation=sum(len(set(g['format_values']))>1 for g in groups.values()),reward_group_variation=sum(max(g['rewards'])>min(g['rewards']) for g in groups.values()),counts=counts,note='Exactly one update from original policy on the same saved192on-policy trajectories; token coefficient0.50. Counts are reused collection costs, not fresh generation. No checkpoint or coefficient selection. Frozen reader; final independent dev rollout still required.')
  save(out/'summary.json',summary);status('complete');print(json.dumps(summary,indent=2))
 except BaseException as e:status('failed',error=repr(e));raise
 finally:
  if embed:embed.close()
if __name__=='__main__':main()
