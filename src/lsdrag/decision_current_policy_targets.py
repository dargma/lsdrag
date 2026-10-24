"""Categorical policy improvement from terminal outcomes under the current policy."""
from collections import Counter
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp

def current_policy_targets(records):
 counts=Counter(r['qid'] for r in records);weights=np.array([1/len(counts)/counts[r['qid']] for r in records])
 entries=[];upper=[];anchors=[]
 for i,r in enumerate(records):
  samples={s['action']:s for s in r['samples']};legal=np.flatnonzero(np.asarray(r['legal']));assert set(samples)==set(legal) and len(samples)==len(r['samples'])
  anchor=int(np.argmax(r['reference_probs']));assert anchor in samples;anchors.append(anchor);ref=samples[anchor]
  failed=any(s['errors'] for s in samples.values())
  for a in legal:
   s=samples[a]
   permitted=a==anchor or (not failed and s['f1']>=ref['f1'] and s['f1']>0)
   entries.append((i,int(a),s['f1'],s['generated']));upper.append(int(permitted))
 n=len(entries);one=np.zeros((len(records),n));q=np.zeros(n);cost=np.zeros(n);changed=np.zeros(n)
 for j,(i,a,f,c) in enumerate(entries):one[i,j]=1;q[j]=weights[i]*f;cost[j]=weights[i]*c;changed[j]=weights[i]*int(a!=anchors[i])
 reference=np.array([int(a==anchors[i]) for i,a,_,_ in entries]);budget=float(cost@reference);baseline_q=float(q@reference)
 rows=[*one,cost];low=[*np.ones(len(records)),-np.inf];high=[*np.ones(len(records)),budget+1e-8]
 def solve(objective):
  result=milp(objective,integrality=np.ones(n),bounds=Bounds(np.zeros(n),upper),constraints=LinearConstraint(np.array(rows),low,high),options={'time_limit':30})
  assert result.success,result.message
  x=np.rint(result.x).astype(int);assert np.max(abs(result.x-x))<1e-5;return x
 x=solve(-q);rows.append(q);low.append(float(q@x)-1e-8);high.append(np.inf)
 x=solve(cost);high[len(records)]=float(cost@x)+1e-8
 x=solve(changed)
 assert cost@x<=budget+1e-6 and q@x>=baseline_q-1e-7
 labels={};moves=[]
 for j,(i,a,f,c) in enumerate(entries):
  if x[j] and a!=anchors[i]:
   ref=next(s for s in records[i]['samples'] if s['action']==anchors[i]);assert f>=ref['f1'] and f>0
   labels[i]=a;moves.append(dict(qid=records[i]['qid'],turn=records[i]['turn'],before=anchors[i],after=a,f1_delta=f-ref['f1'],generated_delta=c-ref['generated']))
 evidence=dict(states=len(records),questions=len(counts),changed_states=len(labels),changed_questions=len({records[i]['qid'] for i in labels}),reference_suffix_f1=baseline_q,target_suffix_f1=float(q@x),reference_suffix_generated=budget,target_suffix_generated=float(cost@x),moves=moves,note='Training hindsight categorical targets: per-state F1 nondecrease, aggregate suffix cost cap; not episode performance or learned-policy guarantee.')
 return labels,evidence
