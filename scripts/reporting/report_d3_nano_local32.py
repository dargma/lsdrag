"""Audit external decision requests and lookup measured H3-suffix outcomes."""
import hashlib,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path('/content/local_fast/lsdrag-text-v1/results');REPO=Path('/content/local_fast/lsdrag-code');NAME='decision_d3_nano_local32_20261024'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=ROOT/NAME;assert read(out/'status.json')['phase']=='complete';m=read(REPO/'lab/registry'/NAME/'manifest.json')
 assert sha(Path(m['cases']))==m['cases_sha256']
 for f,h in m['input_sha256'].items():assert sha(Path(f))==h
 for f,h in m['source_sha256'].items():assert sha(REPO/f)==h
 cases=read(out/'cases.json');pred=read(out/'predictions.json');assert len(cases)==32 and len(pred)==36
 by={(p['qid'],p['order']):p for p in pred};assert len(by)==36
 actions=['keyword_search','semantic_search','read_chunk','FINAL'];rows=[];flips=0;counts=Counter();saved=defaultdict(int)
 for i,c in enumerate(cases):
  req=c['request'];p=by[c['qid'],'canonical'];assert p['request_sha256']==hashlib.sha256(json.dumps(req,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
  choice=p['response']['answers']['next']['choice'];assert choice in req['questions']['next']['criteria'];a=actions.index(choice);counts[choice]+=1
  samples={s['action']:s for s in c['samples']};b=samples[c['reference_action']];s=samples[a];assert a in samples
  group='quality_gain' if s['f1']>b['f1'] else 'quality_loss' if s['f1']<b['f1'] else 'positive_retained' if b['f1']>0 else 'both_zero';saved[group]+=b['generated']-s['generated']
  rows.append(dict(qid=c['qid'],turn=c['turn'],h3_action=c['reference_action'],d3_action=a,h3_f1=b['f1'],d3_f1=s['f1'],h3_generated=b['generated'],d3_generated=s['generated'],h3_errors=b['errors'],d3_errors=s['errors'],d3_regret=max(v['f1'] for v in samples.values())-s['f1']))
  if i<4:
   rev=json.loads(json.dumps(req));rev['questions']['next']['criteria']=dict(reversed(list(req['questions']['next']['criteria'].items())))
   rp=by[c['qid'],'reversed'];assert rp['request_sha256']==hashlib.sha256(json.dumps(rev,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
   flips+=rp['response']['answers']['next']['choice']!=choice
 metrics={name:{k:float(np.mean([r[name+'_'+k] for r in rows])) for k in ['f1','generated']} for name in ['h3','d3']}
 errors=sum(bool(r['d3_errors']) for r in rows);useful=sum(r['d3_f1']>r['h3_f1'] or (r['d3_f1']==r['h3_f1'] and r['d3_f1']>0 and r['d3_generated']<r['h3_generated']) for r in rows)
 gate=metrics['d3']['f1']>=metrics['h3']['f1'] and metrics['d3']['generated']<=metrics['h3']['generated'] and useful>0 and errors==0
 report=dict(verified=True,n=32,requests=36,metrics=metrics,action_counts=dict(counts),h3_action_agreement=sum(r['h3_action']==r['d3_action'] for r in rows)/32,mean_d3_regret=float(np.mean([r['d3_regret'] for r in rows])),quality_wins=sum(r['d3_f1']>r['h3_f1'] for r in rows),quality_losses=sum(r['d3_f1']<r['h3_f1'] for r in rows),savings=dict(saved),permutation_flips=flips,permutation_n=4,selected_branch_errors=errors,diagnostic_gate=gate,runtime=read(out/'runtime.json'),scope='One intervention on32 training states followed by frozenH3; suffix costs overlap no fullpolicy or independenttest claim')
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');dest=REPO/'lab/reports'/NAME;dest.mkdir(parents=True,exist_ok=True)
 for name,obj in [('summary',report),('paired_outcomes',rows)]: (dest/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
