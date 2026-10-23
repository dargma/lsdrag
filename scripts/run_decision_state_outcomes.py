"""Collect same-public-state all-action suffix outcomes; never on-policy GRPO data."""
import argparse,copy,fcntl,hashlib,json,os,re,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from run_decision_policy_pilot import ROOT,REPO,PARENT,sha,save
from lsdrag.decision_policy import PolicyHead,rollout
from lsdrag.decision_policy_recovery import RecoveryPolicyReader
from lsdrag.decision_policy_followup import state_snapshots
from lsdrag.text_integrated import ACTIONS,options_for
from lsdrag.text_decision_benchmark import public_paragraphs
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
REG=REPO/'lab/registry/decision_state_outcomes_20261023'
OUT=ROOT/'results/decision_state_outcomes_20261023'
BASE=ROOT/'results/decision_grpo96_20261023'
def read(p):return json.loads(p.read_text())
def prepare():
 old=read(BASE/'manifest.json');exposed=set()
 for folder in [REPO/'lab',ROOT/'results']:
  for p in folder.rglob('*.json'):
   exposed.update(re.findall(r'\b[234]hop__\d+_\d+\b',p.read_text()))
 data={n:[json.loads(l) for l in (ROOT/f'source/musique/musique_ans_v1.0_{n}.jsonl').open()] for n in ['train','dev']}
 excluded_f={d['id'] for q in data['train']+data['dev'] if q['id'] in exposed for d in q['question_decomposition']}
 fresh=[];used=set()
 for q in sorted(data['train']+data['dev'],key=lambda q:sha_text('eight-hour-state:'+q['id'])):
  f={d['id'] for d in q['question_decomposition']}
  if q['answerable'] and len(f)==2 and q['id'] not in exposed and not f&(excluded_f|used):fresh.append(q['id']);used.update(f)
  if len(fresh)==192:break
 assert len(fresh)==192,('Insufficient fresh families',len(fresh))
 source=['scripts/run_decision_state_outcomes.py','src/lsdrag/decision_policy.py','src/lsdrag/decision_policy_recovery.py','src/lsdrag/decision_policy_followup.py','src/lsdrag/text_integrated.py','src/lsdrag/text_integrated_aligned.py','prompts/text_rag_integrated_v5.txt']
 inputs=[BASE/'grpo_iteration2.pt']+[ROOT/f'source/musique/musique_ans_v1.0_{n}.jsonl' for n in ['train','dev']]
 m={'seed':20261043,'train':old['train'],'validation':fresh[:32],'test':fresh[32:],'model':old['model'],'reference':str(inputs[0]),'states_per_question':2,'selection':'first and last actual controllable state; public trajectory only','repeats_per_action':2,'suffix_policy':'frozen GRPO96 stochastic epsilon0; first action intervention','budget_episodes':1632,'max_seconds':7200,'max_generated':1200000,'max_inputs':40000000,'global_deadline_utc':'2026-10-23T22:48:00Z','source_sha256':{p:sha(REPO/p) for p in source},'input_sha256':{str(p):sha(p) for p in inputs},'target':{'f1_delta':.03,'generated_ratio':.85},'evaluation_source':'fresh unexposed MuSiQue train+dev source, decomposition-family-disjoint; not official dev-only benchmark','test_frozen_before_outcomes':True,'no_test_access_during_collection':True}
 assert not (REG/'manifest.json').exists();save(REG/'manifest.json',m);print('Frozen train96 / validation32 / test160, family-disjoint fresh splits')
def sha_text(x):return hashlib.sha256(x.encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
 if args.prepare:return prepare()
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
  reader=RecoveryPolicyReader(m['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384));embed=V11ProcessEmbedder();head=PolicyHead().cuda();head.load_state_dict(torch.load(m['reference'],weights_only=True,map_location='cuda'));head.eval()
  deadline=time.perf_counter()+m['max_seconds'];prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text();records=[]
  for qi,qid in enumerate(m['train']):
   q=raw[qid];rows=public_paragraphs(q);p=PARENT/'index'/f'{qid}.npy';vec=np.load(p) if p.exists() else embed.encode([r['text'] for r in rows],'document',titles=[r['title'] for r in rows])
   def episode(tag,initial=None,first=None,seed=0):
    assert counts['episodes']<m['budget_episodes'] and counts['generated']+4608<m['max_generated'] and counts['inputs']+148032<m['max_inputs']
    torch.manual_seed(seed);ep,steps,_=rollout(reader,q['question'],rows,vec,embed,prompt,head,deadline,stochastic=True,epsilon=0.,initial=initial,first_action=first)
    ep.update(qid=qid,tag=tag,first_action=first,seed=seed,f1=max(compute_f1(a,ep['answer']) for a in [q['answer']]+q['answer_aliases']))
    save(OUT/'episodes'/f'{qid}_{tag}.json',ep);counts['episodes']+=1;counts['generated']+=ep['generated_tokens'];counts['inputs']+=ep['prompt_tokens'];return ep,steps
   collector,cs=episode('collector',seed=m['seed']+qi*1000)
   snaps=[s for s in state_snapshots(q['question'],prompt,collector['trajectory']) if s['turn']<len(cs) and not cs[s['turn']]['forced']]
   snaps=snaps[:1]+snaps[-1:] if len(snaps)>1 else snaps
   prefills=[u for u in collector['reader_calls'] if u.get('pack_slots')]
   for snap in snaps:
    turn=snap['turn'];samples=[];prefix_hash=prefills[turn]['messages_sha256'];pack=cs[turn]['pack'];legal=cs[turn]['legal']
    for ai in range(4):
     if not bool(legal[ai]):continue
     for rep in range(m['repeats_per_action']):
      ep,steps=episode(f't{turn}_a{ai}_r{rep}',snap,ai,m['seed']+qi*1000+turn*10+rep+1)
      assert ep['reader_calls'][0]['messages_sha256']==prefix_hash,'prefix reconstruction mismatch'
      assert steps[0]['action']==ai and steps[0]['branch_intervention']
      assert torch.equal(steps[0]['legal'],legal)
      assert torch.allclose(steps[0]['pack'].float(),pack.float(),atol=.04,rtol=.02),'same prefix hidden drift'
      samples.append({'action':ai,'rep':rep,'f1':ep['f1'],'cost':ep['cost_units'],'generated':ep['generated_tokens'],'inputs':ep['prompt_tokens'],'errors':ep['errors']})
    records.append({'qid':qid,'turn':turn,'pack':pack,'legal':legal,'prefix_sha256':prefix_hash,'reference_probs':cs[turn]['policy_probs'],'samples':samples})
    counts['states']+=1;torch.save(records,OUT/'states.pt');phase('collection',questions=qi+1)
   phase('collection',questions=qi+1)
  save(OUT/'summary.json',{'counts':counts,'questions':len(m['train']),'states':len(records),'state_tensor_sha256':sha(OUT/'states.pt'),'validation_evaluated':False,'test_evaluated':False});phase('complete')
 except BaseException as e:phase('failed',error=repr(e));raise
 finally:
  if embed:embed.close()
if __name__=='__main__':main()
