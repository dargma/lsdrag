"""Matched frozen-base vs full-head continuation on exposed validation64."""
import json,hashlib
from pathlib import Path
import numpy as np
from compare_decision_policy_campaign import load,ROOT,REPO

def main():
 oldname='decision_structure_attention_validation64_20261024'
 newname='decision_frozen_adapters_validation64_20261024'
 old,oe=load(oldname);new,ne=load(newname)
 for k in ['qids','model','corpus','document_sha256','vectors_sha256']:assert old[k]==new[k]
 keys=['answer','support_ids','f1','generated_tokens','prompt_tokens','tool_calls','errors']
 assert all(all(oe['selected_reference',q][k]==ne['selected_reference',q][k] for k in keys) for q in new['qids'])
 draws=np.random.default_rng(20261063).integers(64,size=(10000,64));result={}
 for variant in ['depth','question_attention','action_attention']:
  a=[ne[variant+'_frozen',q] for q in new['qids']];b=[oe[variant,q] for q in new['qids']]
  diff=np.array([x['f1']-y['f1'] for x,y in zip(a,b)]);ta=np.array([x['generated_tokens'] for x in a]);tb=np.array([x['generated_tokens'] for x in b])
  result[variant]={'f1_delta':float(diff.mean()),'f1_delta_ci95':np.quantile(diff[draws].mean(1),[.025,.975]).tolist(),'generated_ratio':float(ta.mean()/tb.mean()),'generated_ratio_ci95':np.quantile(ta[draws].mean(1)/tb[draws].mean(1),[.025,.975]).tolist()}
 path=REPO/'lab/reports'/newname
 report={'comparisons_frozen_minus_full':result,'reference_exact_parity':64,'scope':'Paired exposed validation64, single seed, marginal bootstrap intervals, no multiplicity correction; not independent confirmation.','goal_achieved':False}
 (path/'frozen_vs_full.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
