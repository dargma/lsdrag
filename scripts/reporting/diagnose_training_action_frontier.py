"""Empirical root-action envelope: hindsight over train-only sampled suffixes."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog
from evaluate_decision_attention import load_head
ROOT=Path('/content/local_fast/lsdrag-text-v1/results')
def main():
 path=ROOT/'decision_selected_refresh48_20261024/states.pt'
 records=[r for r in torch.load(path,weights_only=True,map_location='cpu') if r['turn']==0]
 assert len(records)==len({r['qid'] for r in records})==48
 spec=json.loads(Path('lab/registry/decision_selected_refresh48_20261024/manifest.json').read_text())
 assert {r['qid'] for r in records}==set(spec['train'])
 assert not set(spec['train'])&set(spec['validation']+spec['test'])
 torch.set_num_threads(2);head=load_head(spec['reference_spec']);x=torch.stack([r['pack'] for r in records]).float();mask=torch.stack([r['legal'] for r in records])
 with torch.no_grad():actions=head(x,mask)[0].argmax(-1).tolist()
 cells=[]
 for i,r in enumerate(records):
  for a in range(4):
   ss=[s for s in r['samples'] if s['action']==a]
   if not bool(r['legal'][a]):assert not ss;continue
   assert len(ss)==2
   cells.append(dict(question=i,action=a,quality=np.mean([s['f1'] for s in ss]),tokens=np.mean([s['generated'] for s in ss]),worst_quality=min(s['f1'] for s in ss),worst_tokens=max(s['generated'] for s in ss)))
 eq=np.zeros((48,len(cells)))
 for j,c in enumerate(cells):eq[c['question'],j]=1
 answer={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'questions':48,'limitations':['Hindsight selection using TRAIN outcomes; not deployable performance or independent evaluation.','LP allows randomized action mixtures. Only enumerated root interventions, fixed d987 stochastic suffix policy, two samples/action. Not a bound on all possible RAG policies.','Worst-of-two is not a statistical confidence bound. Errors retained in empirical outcomes.','Train48 differs from validation64: the numeric validation token target is a reference budget only, not an achievement test.']}
 chosen=[c for c in cells if c['action']==actions[c['question']]]
 answer['anchor_action_sampled_suffix_means']={k:float(np.mean([c[k] for c in chosen])) for k in ['quality','tokens']}
 answer['envelopes']={}
 for mode,qkey,ckey in [('mean','quality','tokens'),('worst_of_two','worst_quality','worst_tokens')]:
  q=np.array([c[qkey] for c in cells]);cost=np.array([c[ckey] for c in cells]);curve=[]
  for budget in [50.,60.,68.884375,76.75,81.546875,98.40625]:
   fit=linprog(-q,A_ub=cost[None],b_ub=[48*budget],A_eq=eq,b_eq=np.ones(48),bounds=(0,1),method='highs')
   row={'token_budget':budget,'feasible':bool(fit.success)}
   if fit.success:row.update(expected_f1=float(q@fit.x/48),expected_tokens=float(cost@fit.x/48),fractional_cells=int(((fit.x>1e-7)&(fit.x<1-1e-7)).sum()))
   curve.append(row)
  answer['envelopes'][mode]=curve
 out=Path('lab/reports/decision_selected_refresh48_20261024/training_action_frontier.json');out.write_text(json.dumps(answer,indent=2)+'\n');print(json.dumps(answer,indent=2))
if __name__=='__main__':main()
