"""Fresh paired evaluation of global-bias, STOP and tool-only PPO projections."""
import argparse,json,hashlib,re,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from lsdrag.decision_policy import PolicyHead,PolicyReader,rollout
from lsdrag.decision_policy_ablation import ProjectedHead,fit_global_bias
from lsdrag.text_decision_benchmark import public_paragraphs
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
from run_decision_policy_pilot import ROOT,REPO,PARENT,sha,save
OLD=ROOT/'results/decision_policy_20261023_pilot2';RUNS=[ROOT/'results/decision_ppo_tempered_20261023',ROOT/'results/decision_ppo_tempered_seed2_20261023'];REG=REPO/'lab/registry/decision_policy_ablation_20261023';OUT=ROOT/'results/decision_policy_ablation_20261023'
def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
 if a.prepare:
  old=read(RUNS[0]/'manifest.json');exposed=set()
  for folder in [REPO/'lab',ROOT/'results']:
   for p in folder.rglob('*.json'):exposed.update(re.findall(r'\b[234]hop__\d+_\d+\b',p.read_text()))
  data={s:[json.loads(x) for x in (ROOT/f'source/musique/musique_ans_v1.0_{s}.jsonl').read_text().splitlines()] for s in ['train','dev']};families={d['id'] for q in data['train']+data['dev'] if q['id'] in exposed for d in q['question_decomposition']};test=[];used=set()
  for q in sorted(data['dev'],key=lambda q:hashlib.sha256(('policy-ablation:'+q['id']).encode()).hexdigest()):
   f={d['id'] for d in q['question_decomposition']}
   if q['answerable'] and len(f)==2 and q['id'] not in exposed and not f&(families|used):test.append(q['id']);used.update(f)
   if len(test)==40:break
  assert len(test)==40
  inputs=[OLD/'reference_round0.pt',OLD/'round0/counterfactual_data.pt']+[r/'ppo_iteration2.pt' for r in RUNS];source=['scripts/run_decision_policy_ablation.py','src/lsdrag/decision_policy_ablation.py','src/lsdrag/decision_policy.py','src/lsdrag/text_integrated.py','src/lsdrag/text_integrated_aligned.py','prompts/text_rag_integrated_v5.txt']
  m={'seed':20261036,'test':test,'train':old['train'],'model':old['model'],'policies':['anchor']+[f'{mode}_s{s}' for s in [1,2] for mode in ['full','global_bias','stop_only','tool_only']],'bias_fit':'67 round0 train states, question weighted KL, Adam .05, 200steps, .001 mean bias square, zero-mean4bias','budget_episodes':360,'seconds':1800,'generated_cap':500000,'input_cap':16000000,'primary':'full minus global_bias, average two fixed heads over40paired questions','secondary':'full minus stop_only/tool_only/anchor','source_sha256':{p:sha(REPO/p) for p in source},'input_sha256':{str(p):sha(p) for p in inputs},'scope':'post-hoc mechanism hypothesis, prospective fresh40 evaluation; projection rather than equal-budget outcome retraining'}
  assert not (REG/'manifest.json').exists();save(REG/'manifest.json',m);print('Frozen fresh40 x9 =360 episodes');return
 m=read(REG/'manifest.json');assert all(sha(REPO/p)==h for p,h in m['source_sha256'].items());assert all(sha(Path(p))==h for p,h in m['input_sha256'].items())
 import fcntl
 lock=(ROOT/'decision_policy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(parents=True,exist_ok=False);save(OUT/'manifest.json',m);torch.set_num_threads(2);torch.manual_seed(m['seed']);start=time.perf_counter();count={'episodes':0,'generated':0,'inputs':0};embed=None
 def phase(name,**kw):save(OUT/'status.json',{'phase':name,'pid':None if name in ['complete','failed'] else os.getpid(),'seconds':time.perf_counter()-start,**count,**kw})
 try:
  phase('fitting_bias');anchor=PolicyHead(2560,'mlp');anchor.load_state_dict(torch.load(OLD/'reference_round0.pt',weights_only=True,map_location='cpu'));anchor.eval();heads={'anchor':anchor};stats={}
  train=torch.load(OLD/'round0/counterfactual_data.pt',weights_only=True,map_location='cpu');assert set(r['qid'] for r in train)==set(m['train']);x=torch.stack([r['pack'] for r in train]).float();legal=torch.stack([r['legal'] for r in train]);qids=[r['qid'] for r in train]
  for s,run in enumerate(RUNS,1):
   full=PolicyHead(2560,'mlp');full.load_state_dict(torch.load(run/'ppo_iteration2.pt',weights_only=True,map_location='cpu'));full.eval();heads[f'full_s{s}']=full;bias,stat=fit_global_bias(anchor,full,x,legal,qids);stats[f's{s}']=stat;torch.save(bias,OUT/f'global_bias_s{s}.pt')
   for mode in ['global_bias','stop_only','tool_only']:heads[f'{mode}_s{s}']=ProjectedHead(anchor,full,mode,bias).eval()
   with torch.no_grad():
    al=anchor(x,legal)[0];fl=full(x,legal)[0];sl=heads[f'stop_only_s{s}'](x,legal)[0];tl=heads[f'tool_only_s{s}'](x,legal)[0]
    assert torch.equal(sl[:,:3].argmax(-1),al[:,:3].argmax(-1));assert torch.equal(tl[:,:3].argmax(-1),fl[:,:3].argmax(-1));assert torch.equal(sl.argmax(-1)==3,fl.argmax(-1)==3);assert torch.equal(tl.argmax(-1)==3,al.argmax(-1)==3)
  save(OUT/'fit_stats.json',stats);phase('loading');reader=PolicyReader(m['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384));embed=V11ProcessEmbedder();heads={k:h.cuda() for k,h in heads.items()};deadline=time.perf_counter()+m['seconds'];prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text()
  raw={q['id']:q for q in map(json.loads,(ROOT/'source/musique/musique_ans_v1.0_dev.jsonl').read_text().splitlines()) if q['id'] in set(m['test'])};sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1
  results=[]
  for qi,qid in enumerate(m['test']):
   q=raw[qid];rows=public_paragraphs(q);p=PARENT/'index'/f'{qid}.npy';vec=np.load(p) if p.exists() else embed.encode([r['text'] for r in rows],'document',titles=[r['title'] for r in rows]);order=m['policies'][qi%9:]+m['policies'][:qi%9]
   for name in order:
    assert time.perf_counter()<deadline and count['episodes']<360 and count['generated']+4608<=m['generated_cap'] and count['inputs']+148032<=m['input_cap']
    ep,_,_=rollout(reader,q['question'],rows,vec,embed,prompt,heads[name],deadline,epsilon=0.);ep.update(qid=qid,policy=name,f1=max(compute_f1(a,ep['answer']) for a in [q['answer']]+q['answer_aliases']));save(OUT/f'episodes/{name}_{qid}.json',ep);results.append(ep);count['episodes']+=1;count['generated']+=ep['generated_tokens'];count['inputs']+=ep['prompt_tokens']
   phase('evaluation',questions=qi+1)
  summary={}
  for name in m['policies']:
   es=[e for e in results if e['policy']==name];summary[name]={k:float(np.mean([e[k] for e in es])) for k in ['f1','tool_calls','generated_tokens','prompt_tokens','cost_units']};summary[name].update(utility=summary[name]['f1']-.03*summary[name]['cost_units'],errors=sum(bool(e['errors']) for e in es),forced_final=sum(e['forced_final'] for e in es))
  comparisons={};rng=np.random.default_rng(m['seed']);draw=rng.integers(40,size=(4000,40));lookup={(e['policy'],e['qid']):e for e in results}
  for other in ['global_bias','stop_only','tool_only','anchor']:
   comparisons[other]={}
   for metric in ['f1','utility','tool_calls']:
    def val(e):return e['f1']-.03*e['cost_units'] if metric=='utility' else e[metric]
    delta=np.array([[val(lookup[f'full_s{s}',q])-val(lookup['anchor' if other=='anchor' else f'{other}_s{s}',q]) for q in m['test']] for s in [1,2]]);avg=delta.mean(0);comparisons[other][metric]={'mean':float(avg.mean()),'seed_deltas':delta.mean(1).tolist(),'ci95_conditional_questions':np.quantile(avg[draw].mean(1),[.025,.975]).tolist()}
  save(OUT/'summary.json',{'policies':summary,'full_minus':comparisons,'fit':stats,'counts':count,'test_questions':40,'controlled_latency':False});phase('complete')
 except BaseException as e:phase('failed',error=repr(e));raise
 finally:
  if embed:embed.close()
if __name__=='__main__':main()
