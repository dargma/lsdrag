"""Evaluate a pre-frozen named candidate set; validation selection is separate."""
import argparse,fcntl,json,os,sys,time
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from run_decision_policy_pilot import ROOT,REPO,PARENT,save,sha
from fit_decision_state_heads import load_head
from lsdrag.decision_policy import rollout
from lsdrag.decision_policy_recovery import RecoveryPolicyReader
from lsdrag.text_decision_benchmark import public_paragraphs
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
DATA=ROOT/'results/decision_state_policy_iteration_20261023';HEADS=ROOT/'results/decision_state_heads_20261023'
def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);args=ap.parse_args();spec=read(Path(args.spec));m=read(DATA/'manifest.json');hm=read(HEADS/'manifest.json');out=ROOT/'results'/spec['run_name']
 assert spec['split'] in ['validation','test'];qids=m[spec['split']]
 assert spec['qids']==qids
 heads={}
 for name in spec['candidates']:
  desc=hm['candidates'][name];assert sha(Path(desc['checkpoint']))==spec['head_sha256'][name];heads[name]=load_head(desc,'cuda')
 lock=(ROOT/'decision_policy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out.mkdir(parents=True,exist_ok=False);save(out/'spec.json',spec)
 torch.set_num_threads(2);start=time.monotonic();counts={'episodes':0,'generated':0,'inputs':0};embed=None
 def phase(name,**kw):save(out/'status.json',dict(phase=name,pid=None if name in ['complete','failed'] else os.getpid(),seconds=time.monotonic()-start,**counts,**kw))
 phase('loading');raw={}
 for split in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
   q=json.loads(line)
   if q['id'] in qids:raw[q['id']]=q
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1
 try:
  reader=RecoveryPolicyReader(m['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384));embed=V11ProcessEmbedder();deadline=time.perf_counter()+spec.get('max_seconds',3600);prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text();episodes=[]
  for qi,qid in enumerate(qids):
   q=raw[qid];rows=public_paragraphs(q);p=PARENT/'index'/f'{qid}.npy';vec=np.load(p) if p.exists() else embed.encode([r['text'] for r in rows],'document',titles=[r['title'] for r in rows])
   for name,h in heads.items():
    assert counts['generated']+4608<spec.get('max_generated',1200000) and counts['inputs']+148032<spec.get('max_inputs',40000000)
    ep,_,_=rollout(reader,q['question'],rows,vec,embed,prompt,h,deadline,stochastic=False,epsilon=0.)
    ep.update(qid=qid,policy=name,f1=max(compute_f1(a,ep['answer']) for a in [q['answer']]+q['answer_aliases']));save(out/'episodes'/f'{name}_{qid}.json',ep);episodes.append(ep);counts['episodes']+=1;counts['generated']+=ep['generated_tokens'];counts['inputs']+=ep['prompt_tokens']
   phase('evaluation',questions=qi+1)
  summary={}
  for name in heads:
   eps=[e for e in episodes if e['policy']==name];summary[name]={k:float(np.mean([e[k] for e in eps])) for k in ['f1','cost_units','generated_tokens','prompt_tokens','tool_calls']};summary[name]['errors']=sum(bool(e['errors']) for e in eps);summary[name]['utility']=summary[name]['f1']-.03*summary[name]['cost_units']
  save(out/'summary.json',{'policies':summary,'counts':counts,'split':spec['split'],'n':len(qids),'controlled_latency':False});phase('complete');print(json.dumps(summary,indent=2))
 except BaseException as e:phase('failed',error=repr(e));raise
 finally:
  if embed:embed.close()
if __name__=='__main__':main()
