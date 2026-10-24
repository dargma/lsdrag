"""Collect deterministic FINAL versus best H1 continuation outcomes at public states."""
import argparse,copy,fcntl,hashlib,json,os,re,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from run_decision_policy_pilot import ROOT,REPO,PARENT,sha,save
from lsdrag.decision_policy import PolicyHead,rollout
from evaluate_decision_value_heads import load_head
from lsdrag.decision_policy_recovery import RecoveryPolicyReader
from lsdrag.decision_policy_followup import state_snapshots
from lsdrag.text_integrated import ACTIONS,options_for
from lsdrag.text_decision_benchmark import public_paragraphs
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
REG=REPO/'lab/registry/decision_stop_value_h2_collect48_20261024'
OUT=ROOT/'results/decision_stop_value_h2_collect48_20261024'
BASE=ROOT/'results/decision_grpo96_20261023'
def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
 if args.prepare:raise RuntimeError('Use separately frozen expanded split')
 m=read(REG/'manifest.json');assert all(sha(REPO/p)==h for p,h in m['source_sha256'].items());assert all(sha(Path(p))==h for p,h in m['input_sha256'].items())
 lock=(ROOT/'decision_policy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 OUT.mkdir(parents=True,exist_ok=False);save(OUT/'manifest.json',m);torch.set_num_threads(2)
 start=time.monotonic();counts={'episodes':0,'generated':0,'inputs':0,'states':0};embed=None
 def phase(p,**kw):save(OUT/'status.json',dict(phase=p,pid=None if p in ['complete','failed'] else os.getpid(),seconds=time.monotonic()-start,**counts,**kw))
 phase('loading')
 raw={}
 for n in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{n}.jsonl').open():
   q=json.loads(line)
   if q['id'] in m['train']:raw[q['id']]=q
 assert set(raw)==set(m['train'])
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1
 try:
  reader=RecoveryPolicyReader(m['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384));embed=V11ProcessEmbedder();head=load_head(m['reference_spec'],'cuda');head.eval()
  deadline=time.perf_counter()+m['max_seconds'];prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text();records=[];transitions=[]
  rows=[json.loads(line) for line in (Path(m['corpus'])/'documents.jsonl').open()];vec=np.load(Path(m['corpus'])/'vectors.npy',mmap_mode='r')
  for qi,qid in enumerate(m['train']):
   q=raw[qid]
   def episode(tag,initial=None,first=None,seed=0):
    assert counts['episodes']<m['budget_episodes'] and counts['generated']+4608<m['max_generated'] and counts['inputs']+148032<m['max_inputs']
    torch.manual_seed(seed);ep,steps,_=rollout(reader,q['question'],rows,vec,embed,prompt,head,deadline,stochastic=False,epsilon=0.,initial=initial,first_action=first,state_storage_dtype=torch.float32)
    ep.update(qid=qid,tag=tag,first_action=first,seed=seed,f1=max(compute_f1(a,ep['answer']) for a in [q['answer']]+q['answer_aliases']))
    save(OUT/'episodes'/f'{qid}_{tag}.json',ep)
    pair_prefills=[u for u in ep['reader_calls'] if u.get('pack_slots')]
    assert len(pair_prefills)==len(steps)
    counts['episodes']+=1;counts['generated']+=ep['generated_tokens'];counts['inputs']+=ep['prompt_tokens'];return ep,steps
   collector,cs=episode('collector',seed=m['seed']+qi*1000)
   snaps=[s for s in state_snapshots(q['question'],prompt,collector['trajectory']) if s['turn']<len(cs) and not cs[s['turn']]['forced']]
   snaps=snaps[:1]+snaps[-1:] if len(snaps)>1 else snaps
   prefills=[u for u in collector['reader_calls'] if u.get('pack_slots')]
   for snap in snaps:
    turn=snap['turn'];samples=[];prefix_hash=prefills[turn]['messages_sha256'];pack=cs[turn]['pack'];legal=cs[turn]['legal']
    scores=torch.tensor(cs[turn]['policy_probs'])[:3].masked_fill(~legal[:3],-1.)
    assert bool(legal[3]) and bool(legal[:3].any())
    continue_action=int(scores.argmax())
    for ai in [continue_action,3]:
     for rep in range(m['repeats_per_action']):
      ep,steps=episode(f't{turn}_a{ai}_r{rep}',snap,ai,m['seed']+qi*1000+turn*10+rep+1)
      assert ep['reader_calls'][0]['messages_sha256']==prefix_hash,'prefix reconstruction mismatch'
      assert steps[0]['action']==ai and steps[0]['branch_intervention']
      assert torch.equal(steps[0]['legal'],legal)
      assert torch.allclose(steps[0]['pack'].float(),pack.float(),atol=.04,rtol=.02),'same prefix hidden drift'
      samples.append({'action':ai,'rep':rep,'f1':ep['f1'],'cost':ep['cost_units'],'generated':ep['generated_tokens'],'inputs':ep['prompt_tokens'],'errors':ep['errors']})
    records.append({'qid':qid,'turn':turn,'pack':pack,'legal':legal,'prefix_sha256':prefix_hash,'reference_probs':cs[turn]['policy_probs'],'continue_action':continue_action,'samples':samples})
    counts['states']+=1;torch.save(records,OUT/'states.pt');phase('collection',questions=qi+1)
   phase('collection',questions=qi+1)
   if (qi+1)%16==0:torch.save(transitions,OUT/'transitions.pt')
  torch.save(transitions,OUT/'transitions.pt')
  save(OUT/'summary.json',{'counts':counts,'questions':len(m['train']),'states':len(records),'state_tensor_sha256':sha(OUT/'states.pt'),'transitions':len(transitions),'transition_sha256':sha(OUT/'transitions.pt'),'validation_evaluated':False,'test_evaluated':False});phase('complete')
 except BaseException as e:phase('failed',error=repr(e));raise
 finally:
  if embed:embed.close()
if __name__=='__main__':main()
