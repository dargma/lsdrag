"""Fixed-reference, outcome-blind state selection and fresh suffix repeats."""
import argparse,json,hashlib,os,time,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from lsdrag.decision_policy import PolicyHead,PolicyReader,rollout
from lsdrag.decision_policy_followup import state_snapshots
from lsdrag.text_decision_benchmark import public_paragraphs
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
from run_decision_policy_pilot import ROOT,REPO,PARENT,sha,save
OLD=ROOT/'results/decision_policy_20261023_pilot2'
OUT=ROOT/'results/decision_reliability_20261023'
REG=REPO/'lab/registry/decision_reliability_20261023'
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
 old=json.loads((OLD/'manifest.json').read_text())
 if a.prepare:
  # Identity-only deterministic sampling: one state per question; no outcome ranking.
  rows=torch.load(OLD/'round0/counterfactual_data.pt',weights_only=False,map_location='cpu')
  rank=lambda s:hashlib.sha256(('reliability-20261028:'+s).encode()).hexdigest()
  qs=sorted(old['train'],key=rank)[:12]
  chosen=[min([r for r in rows if r['qid']==q],key=lambda r:rank(q+':'+str(r['turn']))) for q in qs]
  sources=['scripts/run_decision_reliability.py','src/lsdrag/decision_policy.py','src/lsdrag/decision_policy_followup.py','src/lsdrag/text_integrated.py','prompts/text_rag_integrated_v5.txt']
  inputs=[OLD/'reference_round0.pt',OLD/'round0/counterfactual_data.pt']+[OLD/f'round0/collectors/reference_{q}.json' for q in qs]
  m={'seed':20261028,'repeats':6,'states':[{'qid':r['qid'],'turn':r['turn'],'legal':r['legal'].tolist()} for r in chosen], 'source_sha256':{s:sha(REPO/s) for s in sources},'input_sha256':{str(s):sha(s) for s in inputs},'model':old['model'],'max_branches':sum(sum(r['legal'].tolist())*6 for r in chosen),'seconds':1800,'generated_cap':300000,'input_cap':12000000,'scope':'train-only fixed reference reliability; no new policy training or independent collection-policy replication'}
  assert not (REG/'manifest.json').exists();save(REG/'manifest.json',m);print(json.dumps(m,indent=2));return
 m=json.loads((REG/'manifest.json').read_text());assert all(sha(REPO/s)==h for s,h in m['source_sha256'].items());assert all(sha(Path(s))==h for s,h in m['input_sha256'].items())
 import fcntl
 lock=(ROOT/'decision_policy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 OUT.mkdir(parents=True,exist_ok=False);save(OUT/'manifest.json',m);torch.set_num_threads(2)
 save(OUT/'status.json',{'phase':'loading','pid':os.getpid()})
 raw={q['id']:q for q in map(json.loads,(ROOT/'source/musique/musique_ans_v1.0_train.jsonl').read_text().splitlines()) if q['id'] in {s['qid'] for s in m['states']}}
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1
 embed=None;records=[];tokens=inputs=0
 try:
  reader=PolicyReader(m['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384));embed=V11ProcessEmbedder();head=PolicyHead(2560,'mlp').cuda();head.load_state_dict(torch.load(OLD/'reference_round0.pt',weights_only=True,map_location='cuda'));head.eval()
  prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text();deadline=time.perf_counter()+m['seconds']
  for i,s in enumerate(m['states']):
   q=raw[s['qid']];collector=json.loads((OLD/f'round0/collectors/reference_{s["qid"]}.json').read_text());snap=next(x for x in state_snapshots(q['question'],prompt,collector['trajectory']) if x['turn']==s['turn'])
   rows=public_paragraphs(q);vp=PARENT/'index'/f'{q["id"]}.npy';vec=np.load(vp) if vp.exists() else embed.encode([r['text'] for r in rows],'document',titles=[r['title'] for r in rows])
   for rep in range(m['repeats']):
    for ai,legal in enumerate(s['legal']):
     if not legal:continue
     assert time.perf_counter()<deadline and tokens+4608<=m['generated_cap'] and inputs+148032<=m['input_cap']
     torch.manual_seed(m['seed']+i*100+rep)
     ep,steps,_=rollout(reader,q['question'],rows,vec,embed,prompt,head,deadline,stochastic=True,epsilon=0.,initial=snap,first_action=ai)
     assert steps[0]['legal'].tolist()==s['legal']
     f1=max(compute_f1(x,ep['answer']) for x in [q['answer']]+q['answer_aliases'])
     ep.update(qid=q['id'],turn=s['turn'],repeat=rep,intervention=ai,f1=f1)
     save(OUT/f'episodes/s{i}_r{rep}_a{ai}.json',ep)
     records.append({'state':i,'qid':q['id'],'turn':s['turn'],'repeat':rep,'action':ai,'f1':f1,'cost':ep['cost_units'],'utility':f1-.03*ep['cost_units']})
     tokens+=ep['generated_tokens'];inputs+=ep['prompt_tokens'];save(OUT/'samples.json',records)
    save(OUT/'status.json',{'phase':'collecting','pid':os.getpid(),'episodes':len(records),'state':i,'repeat':rep,'generated':tokens,'inputs':inputs})
  assert len(records)==m['max_branches'];save(OUT/'status.json',{'phase':'complete','pid':None,'episodes':len(records),'generated':tokens,'inputs':inputs})
 except BaseException as e:
  save(OUT/'status.json',{'phase':'failed','pid':None,'error':repr(e),'episodes':len(records)});raise
 finally:
  if embed:embed.close()
if __name__=='__main__':main()
