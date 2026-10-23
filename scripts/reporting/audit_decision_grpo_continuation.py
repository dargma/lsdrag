"""Audit additional ordinary GRPO control, keeping validation distinct from test."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[2];RUN=ROOT/'results/decision_grpo_continuation_20261023'
def read(p):return json.loads(p.read_text())
def main():
 m=read(RUN/'manifest.json');status=read(RUN/'status.json');s=read(RUN/'summary.json');assert status['phase']=='complete';eps=[read(p) for p in (RUN/'episodes').glob('*.json')];assert len(eps)==m['budget_episodes']==1216
 for name,v in s['policies'].items():
  es=[e for e in eps if e['policy']=='test_'+name];assert {e['qid'] for e in es}==set(m['test']) and len(es)==32
  for k in ['f1','generated_tokens','prompt_tokens','cost_units','tool_calls']:assert np.isclose(v[k],np.mean([e[k] for e in es]))
 for path,digest in read(RUN/'frozen_evaluation.json')['head_sha256'].items():assert hashlib.sha256((RUN/f'{path}_iteration2.pt').read_bytes()).hexdigest()==digest
 for key,field in [('generated','generated_tokens'),('inputs','prompt_tokens')]:assert sum(e[field] for e in eps)==status[key]
 nd=0
 for e in eps:
  for d in e['decisions']:
   p=np.array(d['policy_probs']);legal=np.array(d['legal']);assert np.isfinite(p).all() and abs(p.sum()-1)<1e-5 and (p[~legal]==0).all();nd+=1
 curves=read(RUN/'curves.json');result=s|{'audit_passed':True,'decisions':nd,'optimizer_steps':sum(4*((c['states']+31)//32) for c in curves),'evaluation_role':'validation32, not final test','initialization':'frozen GRPO96 final checkpoint','comparison_limit':'1152 additional online rollouts; not strict token-matched to all-action data collection','curves':curves}
 p=REPO/'lab/reports/decision_grpo_continuation_20261023/result.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='curves'},indent=2))
if __name__=='__main__':main()
