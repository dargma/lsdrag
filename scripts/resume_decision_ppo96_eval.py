"""Resume missing frozen eval only; preserve first crashed cell as failure."""
import json,hashlib,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from lsdrag.decision_policy import PolicyHead,rollout
from lsdrag.decision_policy_recovery import RecoveryPolicyReader
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
from lsdrag.text_decision_benchmark import public_paragraphs
from run_decision_policy_pilot import ROOT,REPO,PARENT,ORIGIN,sha,save
RUN=ROOT/'results/decision_ppo96_20261023';OLD=ROOT/'results/decision_policy_20261023_pilot2'
def read(p):return json.loads(p.read_text())
def main():
 m=read(RUN/'manifest.json');assert all(sha(Path(p))==h for p,h in m['input_sha256'].items());assert all(sha(REPO/p)==h for p,h in m['source_sha256'].items());frozen=read(RUN/'frozen_evaluation.json');assert all(sha(RUN/f'{n}_iteration2.pt')==h for n,h in frozen['head_sha256'].items())
 import fcntl
 lock=(ROOT/'decision_policy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 existing={str(p):sha(p) for p in (RUN/'episodes').glob('*.json')};missing=[(q,n) for q in m['test'] for n in m['policies'] if not (RUN/f'episodes/test_{n}_{q}.json').exists()];assert len(existing)==1222 and len(missing)==90
 recovery=RUN/'recovery';recovery.mkdir(exist_ok=False);save(recovery/'original_failed_status.json',read(RUN/'status.json'));save(recovery/'manifest.json',{'preserved_episode_sha256':existing,'first_failed_cell':missing[0],'missing_cells':missing,'source_sha256':{str(p):sha(REPO/p) for p in ['scripts/resume_decision_ppo96_eval.py','src/lsdrag/decision_policy_recovery.py']},'rule':'First crashed cell scores failure even if replay succeeds. No policy retraining. Original failed attempt cost unavailable; replay cost is proxy.'})
 torch.set_num_threads(2);embed=None;start=time.perf_counter();counts={'episodes':1222,'generated':0,'inputs':0}
 for p in existing:
  e=read(Path(p));counts['generated']+=e['generated_tokens'];counts['inputs']+=e['prompt_tokens']
 def phase(name,**kw):save(RUN/'status.json',{'phase':name,'pid':None if name in ['complete','failed'] else os.getpid(),**counts,'recovery_seconds':time.perf_counter()-start,**kw})
 try:
  phase('recovery_loading');reader=RecoveryPolicyReader(m['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384));embed=V11ProcessEmbedder();deadline=time.perf_counter()+1800;heads={}
  for name in ['tempered_anchor','ppo','rwr']:
   h=PolicyHead(2560,'mlp').cuda();p=OLD/'reference_round0.pt' if name=='tempered_anchor' else RUN/f'{name}_iteration2.pt';h.load_state_dict(torch.load(p,weights_only=True,map_location='cuda'));heads[name]=h.eval()
  hist=torch.load(ORIGIN/'head.pt',weights_only=True,map_location='cuda');historical={k:hist[k] for k in ['action_weight','action_bias']};raw={q['id']:q for q in map(json.loads,(ROOT/'source/musique/musique_ans_v1.0_dev.jsonl').read_text().splitlines()) if q['id'] in set(m['test'])};prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text();cache={};sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1
  for index,(qid,name) in enumerate(missing):
   assert time.perf_counter()<deadline;q=raw[qid]
   if qid not in cache:
    rows=public_paragraphs(q);p=PARENT/'index'/f'{qid}.npy';vec=np.load(p) if p.exists() else embed.encode([r['text'] for r in rows],'document',titles=[r['title'] for r in rows]);cache[qid]=(rows,vec)
   rows,vec=cache[qid];ep,_,_=rollout(reader,q['question'],rows,vec,embed,prompt,heads.get(name),deadline,epsilon=0.,historical=historical if name=='historical_bc' else None)
   ep.update(qid=qid,policy=f'test_{name}',f1=max(compute_f1(a,ep['answer']) for a in [q['answer']]+q['answer_aliases']))
   if index==0:
    save(recovery/'first_failed_cell_replay.json',ep);ep=dict(ep);ep.update(answer='',support_ids=[],f1=0.,errors=list(ep['errors'])+['Original evaluation aborted: Grammar token mismatch; retained failure'],original_failure_retained=True,cost_note='Replay cost proxy; original failed attempt cost not available')
   save(RUN/f'episodes/test_{name}_{qid}.json',ep);counts['episodes']+=1;counts['generated']+=ep['generated_tokens'];counts['inputs']+=ep['prompt_tokens'];phase('recovery_evaluation',completed_missing=index+1)
  assert all(sha(Path(p))==h for p,h in existing.items());evaluation=[read(RUN/f'episodes/test_{name}_{q}.json')|{'condition':name} for q in m['test'] for name in m['policies']];summary={}
  for name in m['policies']:
   es=[e for e in evaluation if e['condition']==name];summary[name]={k:float(np.mean([e[k] for e in es])) for k in ['f1','tool_calls','cost_units','generated_tokens','prompt_tokens']};summary[name].update(errors=sum(bool(e['errors']) for e in es),forced_final=sum(e['forced_final'] for e in es));summary[name]['utility']=summary[name]['f1']-.03*summary[name]['cost_units']
  comp={};rng=np.random.default_rng(m['seed']);draw=rng.integers(40,size=(4000,40))
  for name in ['historical_bc','tempered_anchor','rwr']:
   comp[name]={}
   for metric in ['f1','utility']:
    def value(e):return e['f1'] if metric=='f1' else e['f1']-.03*e['cost_units']
    delta=np.array([value(next(e for e in evaluation if e['qid']==q and e['condition']=='ppo'))-value(next(e for e in evaluation if e['qid']==q and e['condition']==name)) for q in m['test']]);comp[name][metric]={'delta':float(delta.mean()),'ci95':np.quantile(delta[draw].mean(1),[.025,.975]).tolist()}
  save(RUN/'summary.json',{'policies':summary,'ppo_minus':comp,'counts':counts,'single_seed':True,'controlled_latency':False,'recovery':{'preserved_existing':1222,'missing_completed':90,'original_failed_cell':missing[0],'failure_score_retained':True,'original_failed_attempt_extra_cost_unavailable':True,'no_retraining':True}});phase('complete',recovery_completed=True)
 except BaseException as e:phase('failed',error=repr(e));raise
 finally:
  if embed:embed.close()
if __name__=='__main__':main()
