"""Audit collection coverage, state identity and cached counterfactual targets."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/content/local_fast/lsdrag-text-v1');RUN=ROOT/'results/decision_state_outcomes_20261023';REPO=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 status=read(RUN/'status.json');assert status['phase']=='complete';m=read(RUN/'manifest.json');records=torch.load(RUN/'states.pt',weights_only=True);eps=[read(p) for p in (RUN/'episodes').glob('*.json')];assert len(eps)==status['episodes'];assert len(records)==status['states'];assert {r['qid'] for r in records}==set(m['train'])
 for metric,key in [('generated_tokens','generated'),('prompt_tokens','inputs')]:assert sum(e[metric] for e in eps)==status[key]
 bykey={(e['qid'],e['tag']):e for e in eps};assert len(bykey)==len(eps);decision_count=0
 for e in eps:
  assert e['qid'] in m['train']
  for d in e['decisions']:
   p=np.array(d['policy_probs']);legal=np.array(d['legal']);assert np.isfinite(p).all() and abs(p.sum()-1)<1e-5 and (p[~legal]==0).all();decision_count+=1
 sample_counts=[];quality_spreads=[];utility_spreads=[];prefixes=set()
 for r in records:
  assert torch.isfinite(r['pack']).all();assert r['prefix_sha256'] not in prefixes;prefixes.add(r['prefix_sha256'])
  q=[];u=[]
  for a,ok in enumerate(r['legal']):
   rows=[s for s in r['samples'] if s['action']==a];assert len(rows)==(m['repeats_per_action'] if ok else 0)
   for s in rows:
    ep=bykey[(r['qid'],f't{r["turn"]}_a{a}_r{s["rep"]}')]
    assert ep['first_action']==a and ep['decisions'][0]['action']==a and ep['decisions'][0]['branch_intervention']
    assert ep['reader_calls'][0]['messages_sha256']==r['prefix_sha256']
    assert ep['f1']==s['f1'] and ep['cost_units']==s['cost']
   if ok:q.append(np.mean([s['f1'] for s in rows]));u.append(np.mean([s['f1']-.03*s['cost'] for s in rows]))
  quality_spreads.append(max(q)-min(q));utility_spreads.append(max(u)-min(u));sample_counts.append(len(r['samples']))
 wanted=set(m['train']+m['validation']+m['test']);families={};sources={}
 for split in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
   q=json.loads(line)
   if q['id'] in wanted:families[q['id']]={d['id'] for d in q['question_decomposition']};sources[q['id']]=split
 sets={k:set.union(*(families[q] for q in m[k])) for k in ['train','validation','test']}
 assert not (sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test'])
 result={'audit_passed':True,'episodes':len(eps),'decisions':decision_count,'states':len(records),'questions':len(m['train']),'prefixes_unique':len(prefixes),'state_sha256':sha(RUN/'states.pt'),'errors':sum(bool(e['errors']) for e in eps),'forced_final':sum(e['forced_final'] for e in eps),'mean_quality_spread':float(np.mean(quality_spreads)),'mean_utility_spread':float(np.mean(utility_spreads)),'states_quality_spread_positive':int(sum(v>1e-6 for v in quality_spreads)),'states_utility_spread_positive':int(sum(v>1e-6 for v in utility_spreads)),'counts':{k:status[k] for k in ['generated','inputs']},'split_source_counts':{k:{split:sum(sources[q]==split for q in m[k]) for split in ['train','dev']} for k in ['train','validation','test']},'split_family_disjoint':True,'interpretation':'training-state counterfactual target diversity, not heldout policy quality or causal effect under arbitrary suffix policies'}
 p=REPO/'lab/reports/decision_state_outcomes_20261023/audit.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
