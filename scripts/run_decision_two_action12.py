"""24 explicitly scripted search/read interventions, then frozen H3 continuation."""
import fcntl,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from run_decision_policy_pilot import ROOT,REPO,sha,save
from evaluate_decision_stop_value_h2 import load_head
from lsdrag.decision_policy import rollout
from lsdrag.decision_policy_recovery import RecoveryPolicyReader
from lsdrag.decision_prefix_intervention import PrefixIntervention
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
NAME='decision_two_action12_20261024'
def read(p):return json.loads(p.read_text())
def main():
 p=read(REPO/'lab/registry'/NAME/'plan.json')
 for f,h in p['hashes'].items():assert sha(Path(f))==h,f
 lock=(ROOT/'decision_policy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=ROOT/'results'/NAME;out.mkdir(exist_ok=False);save(out/'plan.json',p)
 start=time.monotonic();counts=dict(episodes=0,generated=0,inputs=0);embed=None
 def phase(name,**kw):save(out/'status.json',dict(phase=name,pid=None if name in ['complete','failed'] else os.getpid(),seconds=time.monotonic()-start,**counts,**kw))
 phase('loading');torch.set_num_threads(2);raw={}
 for split in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
   q=json.loads(line)
   if q['id'] in p['qids']:raw[q['id']]=q
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1,compute_exact
 try:
  reader=RecoveryPolicyReader(p['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384));embed=V11ProcessEmbedder();anchor=load_head(p['reference_spec'],'cuda')
  rows=[json.loads(line) for line in (Path(p['corpus'])/'documents.jsonl').open()];vec=np.load(Path(p['corpus'])/'vectors.npy',mmap_mode='r');prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text();deadline=time.perf_counter()+600;features=[]
  for qid in p['qids']:
   q=raw[qid]
   for first in [0,1]:
    wrapper=PrefixIntervention(anchor,[first,2]).eval()
    ep,steps,_=rollout(reader,q['question'],rows,vec,embed,prompt,wrapper,deadline,stochastic=False,epsilon=0.,state_storage_dtype=torch.float32)
    assert len(steps)>=2 and [s['action'] for s in steps[:2]]==[first,2]
    for i,d in enumerate(ep['decisions']):d['scripted_intervention']=i<2;d['h3_reference_probs']=wrapper.reference_probs[i]
    answers=[q['answer']]+q['answer_aliases'];ep.update(qid=qid,first=first,f1=max(compute_f1(a,ep['answer']) for a in answers),em=max(compute_exact(a,ep['answer']) for a in answers),scope='Two scripted actions then H3, not learned-policy evaluation')
    save(out/'episodes'/f'{qid}_a{first}_read.json',ep)
    for i,s in enumerate(steps):
     features.append(dict(qid=qid,first=first,turn=i,pack=s['pack'],legal=s['legal'],action=s['action'],reference_probs=wrapper.reference_probs[i],scripted_intervention=i<2))
    counts['episodes']+=1;counts['generated']+=ep['generated_tokens'];counts['inputs']+=ep['prompt_tokens'];torch.save(features,out/'states.pt');phase('collection')
  assert counts['episodes']==24;phase('complete')
 except BaseException as e:phase('failed',error=repr(e));raise
 finally:
  if embed:embed.close()
if __name__=='__main__':main()
