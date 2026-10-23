#!/usr/bin/env python
"""Frozen step3: fit a STOP-only control and independently evaluate."""
import argparse,copy,hashlib,json,os,re,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from lsdrag.decision_policy import PolicyHead,PolicyReader,rollout
from lsdrag.decision_policy_stop import StopOnlyHead
from lsdrag.decision_policy_followup import fit_improvement
from lsdrag.text_integrated import ACTIONS,run_episode
from lsdrag.text_decision_benchmark import public_paragraphs
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
from run_decision_policy_pilot import ROOT,REPO,PARENT,ORIGIN,sha,save
PREVIOUS=ROOT/'results/decision_policy_20261023_pilot2'
FIRST=ROOT/'results/decision_policy_20261023_pilot1'
REG=REPO/'lab/registry/decision_policy_stop_20261023'
POLICIES=['native','historical_bc','full_mlp_cost','stop_only']

def prepare():
 old=json.loads((PREVIOUS/'manifest.json').read_text());exposed=set()
 for folder in [REPO/'lab',ROOT/'results']:
  for p in folder.rglob('*.json'):exposed.update(re.findall(r'\b[234]hop__\d+_\d+\b',p.read_text()))
 data={n:[json.loads(x) for x in (ROOT/f'source/musique/musique_ans_v1.0_{n}.jsonl').read_text().splitlines()] for n in ['train','dev']}
 families={d['id'] for q in data['train']+data['dev'] if q['id'] in exposed for d in q['question_decomposition']}
 test=[];used=set()
 for q in sorted(data['dev'],key=lambda q:hashlib.sha256(('policy-stop3:'+q['id']).encode()).hexdigest()):
  f={d['id'] for d in q['question_decomposition']}
  if q['answerable'] and len(f)==2 and q['id'] not in exposed and not f&(families|used):test.append(q['id']);used.update(f)
  if len(test)==60:break
 assert len(test)==60
 source=['scripts/run_decision_policy_stop.py','src/lsdrag/decision_policy_stop.py','src/lsdrag/decision_policy_followup.py','src/lsdrag/decision_policy.py','src/lsdrag/text_integrated.py','src/lsdrag/text_integrated_aligned.py','prompts/text_rag_integrated_v5.txt']
 inputs=[PREVIOUS/f'round{i}/counterfactual_data.pt' for i in range(2)]+[PREVIOUS/'heads/mlp_pg_cost_round1.pt',FIRST/'heads/last_mlp.pt',ORIGIN/'head.pt']
 m={'profile':'decision_policy_stop_20261023','seed':20261025,'train':old['train'],'validation':old['validation'],'test':test,'excluded_qids':sorted(exposed),'test_frozen_before_outcomes':True,'model':old['model'],'policies':POLICIES,'training':'reuse exact pilot2 round0 then round1 train-only data; same reference Q targets','objective':'expected-return with fixed non-FINAL BC action; binary residual gate','cost_weight':.03,'temperature':8.,'epochs_per_round':100,'lr':.001,'anchor_KL':.02,'entropy_coef':.01,'single_seed':True,'no_checkpoint_or_threshold_selection':True,'primary':'stop_only vs historical_bc; full_mlp_cost vs stop_only','limits':{'seconds_excluding_load':1800,'generated_tokens':400000,'input_tokens':8000000,'max_tool_calls':8,'max_output_tokens':512,'context_limit':16384},'input_sha256':{str(p):sha(p) for p in inputs},'source_sha256':{p:sha(REPO/p) for p in source}}
 REG.mkdir(parents=True,exist_ok=True)
 if (REG/'manifest.json').exists():raise FileExistsError('frozen manifest exists')
 save(REG/'manifest.json',m);print(REG/'manifest.json')

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
 if a.prepare:prepare();return
 m=json.loads((REG/'manifest.json').read_text());assert all(sha(REPO/p)==h for p,h in m['source_sha256'].items());assert all(sha(Path(p))==h for p,h in m['input_sha256'].items())
 out=a.output;out.mkdir(parents=True,exist_ok=False);save(out/'manifest.json',m)
 import fcntl
 lock=(ROOT/'decision_policy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(2);torch.manual_seed(m['seed']);start=time.perf_counter();embedder=None;counts={'episodes':0,'generated':0,'inputs':0}
 def phase(name,**kw):
  s={'phase':name,'pid':os.getpid(),'elapsed_seconds':time.perf_counter()-start,**counts,**kw};save(out/'status.json',s);print(json.dumps(s),flush=True)
 def charge(ep):
  counts['episodes']+=1;counts['generated']+=ep['generated_tokens'];counts['inputs']+=ep['prompt_tokens']
 def reserve():
  if time.perf_counter()>deadline or counts['generated']+4608>400000 or counts['inputs']+148032>8000000:raise TimeoutError('bounded episode budget')
 try:
  phase('training_stop_only')
  hist=torch.load(ORIGIN/'head.pt',weights_only=True,map_location='cpu');historical={k:hist[k] for k in ['action_weight','action_bias']}
  base=PolicyHead(2560,'mlp');base.load_state_dict(torch.load(FIRST/'heads/last_mlp.pt',weights_only=True,map_location='cpu'))
  anchor=StopOnlyHead(historical,base,m['temperature']).eval();learned=copy.deepcopy(anchor);training=[];parity=0
  for rid in range(2):
   rows=torch.load(PREVIOUS/f'round{rid}/counterfactual_data.pt',weights_only=True,map_location='cpu')
   assert set(r['qid'] for r in rows)==set(m['train'])
   x=torch.stack([r['pack'] for r in rows]).float();legal=torch.stack([r['legal'] for r in rows]);q=torch.zeros(len(rows),4)
   expected=(x[:,0]@historical['action_weight'].T+historical['action_bias']).masked_fill(~legal,-1e9).argmax(-1)
   assert torch.equal(anchor(x,legal)[0].argmax(-1),expected);parity+=len(rows)
   for i,row in enumerate(rows):
    for ai in range(4):
     samples=[s for s in row['samples'] if s['action']==ai]
     if samples:q[i,ai]=np.mean([s['f1']-.03*s['cost'] for s in samples])
   learned,curve=fit_improvement(learned,anchor,x,legal,q,[r['qid'] for r in rows],'return_pg',epochs=100)
   assert torch.equal(learned.tool_weight,anchor.tool_weight) and torch.equal(learned.tool_bias,anchor.tool_bias)
   assert torch.equal(learned(x,legal)[0][:,:3].argmax(-1),(x[:,0]@historical['action_weight'].T+historical['action_bias'])[:,:3].masked_fill(~legal[:,:3],-1e9).argmax(-1))
   torch.save(learned.state_dict(),out/f'stop_only_round{rid}.pt');save(out/f'training_round{rid}.json',curve)
   training.append({'round':rid,'states':len(rows),'data_sha256':sha(PREVIOUS/f'round{rid}/counterfactual_data.pt'),'final_loss':curve[-1]});phase('stop_update_complete',round=rid,states=len(rows))
  save(out/'training_summary.json',{'rounds':training,'initial_BC_argmax_parity':parity,'fixed_tool_selector_verified':True,'trainable_parameters':sum(p.numel() for p in learned.parameters())})
  save(out/'frozen_evaluation.json',{'policies':POLICIES,'test':m['test'],'stop_sha256':sha(out/'stop_only_round1.pt'),'full_head_sha256':sha(PREVIOUS/'heads/mlp_pg_cost_round1.pt'),'no_selection':True})
  phase('loading');reader=PolicyReader(m['model'],SimpleNamespace(max_output_tokens=512,context_limit=16384));embedder=V11ProcessEmbedder();deadline=time.perf_counter()+1800
  learned=learned.cuda();historical={k:v.cuda() for k,v in historical.items()};full=PolicyHead(2560,'mlp').cuda();full.load_state_dict(torch.load(PREVIOUS/'heads/mlp_pg_cost_round1.pt',weights_only=True,map_location='cuda'));full.eval()
  wanted=set(m['validation']+m['test']);raw={}
  for split in ['train','dev']:
   for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
    row=json.loads(line)
    if row['id'] in wanted:raw[row['id']]=row
  sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_exact,compute_f1
  prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text();results=[]
  for group in ['validation','test']:
   for qi,qid in enumerate(m[group]):
    item=raw[qid];rows=public_paragraphs(item);ip=PARENT/'index'/f'{qid}.npy';vec=np.load(ip) if ip.exists() else embedder.encode([r['text'] for r in rows],'document',titles=[r['title'] for r in rows])
    order=POLICIES[qi%4:]+POLICIES[:qi%4]
    for name in order:
     reserve()
     if name=='native':
      ep,_,_=run_episode(item['question'],rows,vec,embedder,reader,prompt,'native','always_generate',deadline);ep.update(generated_tokens=ep['completion_tokens'],forced_prefix_tokens=0);ep['cost_units']=ep['tool_calls']+ep['generated_tokens']/100+ep['prompt_tokens']/10000
     else:
      ep,steps,_=rollout(reader,item['question'],rows,vec,embedder,prompt,learned if name=='stop_only' else full if name=='full_mlp_cost' else None,deadline,epsilon=0.,historical=historical if name=='historical_bc' else None)
      for step,d in zip(steps,ep['decisions']):
       d['selected_action']=ACTIONS[d['action']];d['action_probabilities']=dict(zip(ACTIONS,d['policy_probs']))
       if name=='stop_only' and d['action']!=3:
        old=(historical['action_weight']@step['pack'][0].float().cuda()+historical['action_bias'])[:3].masked_fill(~step['legal'][:3].cuda(),-1e9)
        assert d['action']==int(old.argmax()),'STOP control changed fixed BC tool'
     aliases=[item['answer']]+item['answer_aliases'];gold={f'p{r["idx"]}' for r in item['paragraphs'] if r['is_supporting']};pred=set(ep['support_ids'])
     ep.update(qid=qid,policy=name,group=group,f1=max(compute_f1(x,ep['answer']) for x in aliases),em=max(compute_exact(x,ep['answer']) for x in aliases),support_f1=2*len(gold&pred)/max(1,len(gold)+len(pred)))
     charge(ep);save(out/group/f'{name}_{qid}.json',ep);results.append(ep)
    phase(group+'_evaluation',qid=qid)
  summary={}
  for group in ['validation','test']:
   summary[group]={}
   for name in POLICIES:
    eps=[e for e in results if e['group']==group and e['policy']==name]
    summary[group][name]={'n':len(eps),**{k:float(np.mean([e[k] for e in eps])) for k in ['f1','em','support_f1','tool_calls','generated_tokens','prompt_tokens','forced_prefix_tokens','cost_units','seconds']},'errors':sum(bool(e['errors']) for e in eps),'forced_final':sum(e['forced_final'] for e in eps)}
  pairs={};rng=np.random.default_rng(m['seed'])
  for left,right in [('stop_only','historical_bc'),('full_mlp_cost','stop_only'),('full_mlp_cost','historical_bc')]:
   es={n:{e['qid']:e for e in results if e['policy']==n and e['group']=='test'} for n in [left,right]};d=np.array([es[left][qid]['f1']-es[right][qid]['f1'] for qid in m['test']]);bs=d[rng.integers(0,len(d),size=(4000,len(d)))].mean(-1)
   pairs[left+'_minus_'+right]={'f1_delta':float(d.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist(),'wins':int((d>0).sum()),'losses':int((d<0).sum()),'calls_delta':summary['test'][left]['tool_calls']-summary['test'][right]['tool_calls']}
  save(out/'summary.json',{'policies':summary,'comparisons':pairs,'counts':counts,'single_seed':True,'controlled_latency':False,'reference_Q_not_stop_only_continuation':True,'initialization_differs_from_full_policy':True});phase('complete',training_complete=True,evaluation_complete=True)
 except Exception as e:phase('failed_or_budget_limited',error=repr(e));raise
 finally:
  if embedder:embedder.close()
if __name__=='__main__':main()
