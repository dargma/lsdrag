"""Conditional refresh under learned policy after replication; untouched test stays closed."""
from pathlib import Path
import json,os,time,subprocess,traceback,hashlib
from datetime import datetime,timezone
REPO=Path(__file__).resolve().parents[1];ROOT=Path('/content/local_fast/lsdrag-text-v1');PY=ROOT/'runtime/bin/python';HEADS=ROOT/'results/decision_state_heads_20261023';STATE=ROOT/'results/decision_policy_iteration_coordinator_20261023.json'
def read(p):return json.loads(p.read_text())
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def phase(p,**kw):save(STATE,dict(phase=p,pid=None if p in ['complete','failed'] else os.getpid(),**kw))
def remain():return datetime(2026,10,23,21,48,tzinfo=timezone.utc).timestamp()-time.time()
def run(script,*args):
 if remain()<600:raise TimeoutError('Reporting-hour reserve')
 log=ROOT/'logs'/f'policy_iteration_{Path(script).stem}_{int(time.time())}.log'
 with log.open('w') as f:
  p=subprocess.Popen([str(PY),script,*args],cwd=REPO,env=dict(os.environ,PYTHONPATH='src'),stdout=f,stderr=subprocess.STDOUT)
  phase('running',script=script,child_pid=p.pid,log=str(log))
  try:p.wait(timeout=remain())
  except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30);raise
  if p.returncode:raise RuntimeError(f'{script} failed; {log}')
def main():
 phase('waiting_seed2_replication')
 while True:
  s=read(ROOT/'results/decision_refinement_coordinator_20261023.json')
  if s['phase']=='complete':break
  if s['phase']=='failed':raise RuntimeError('Previous refinement failed')
  p=Path(f'/proc/{s.get("pid")}/cmdline')
  if not p.exists() or b'coordinate_decision_refinement.py' not in p.read_bytes():raise RuntimeError('Previous coordinator disappeared')
  if remain()<600:raise TimeoutError('Deadline')
  time.sleep(15)
 selection=read(HEADS/'final_candidate_selection_before_seed2.json');primary=selection['primary'];v1=selection['all_validation'][primary];base=selection['all_validation']['grpo_baseline'];v2=read(ROOT/'results/decision_state_validation_seed2_20261023/summary.json')['policies'][primary+'_seed2']
 q=(v1['f1']+v2['f1'])/2-base['f1'];ratio=(v1['generated_tokens']+v2['generated_tokens'])/2/base['generated_tokens'];needed=q<.05 or ratio>.7
 decision={'f1_delta_mean':q,'generated_ratio_mean':ratio,'expand':needed,'basis':'validation two-seed mean only; final test unobserved','remaining_seconds':remain()}
 save(REPO/'lab/registry/decision_state_policy_iteration_20261023/trigger.json',decision)
 if not needed:return phase('complete',expanded=False,reason='Validation replication mean meets raised point target; inspect and freeze final test')
 if remain()<9000:return phase('complete',expanded=False,reason='Insufficient time for bounded refresh plus reporting; preserve current candidates')
 run('scripts/run_decision_state_policy_iteration.py');run('scripts/fit_decision_policy_iteration.py')
 m=read(ROOT/'results/decision_state_policy_iteration_20261023/manifest.json');hm=read(HEADS/'manifest.json');cp=ROOT/'results/decision_grpo_continuation_20261023/grpo_iteration2.pt';hm['candidates']['ordinary_grpo_continuation']={'kind':'original','checkpoint':str(cp),'sha256':hashlib.sha256(cp.read_bytes()).hexdigest()};save(HEADS/'manifest.json',hm)
 names=list(dict.fromkeys(['grpo_baseline','ordinary_grpo_continuation',primary,selection['secondary'],'policy_iteration_cost006','policy_iteration_cost009']));names=[n for n in names if n]
 spec={'split':'validation','run_name':'decision_policy_iteration_validation64_20261023','candidates':names,'qids':m['validation'],'head_sha256':{n:hm['candidates'][n]['sha256'] for n in names},'max_seconds':3600,'max_generated':1000000,'max_inputs':30000000}
 path=REPO/'lab/registry/decision_state_policy_iteration_20261023/validation_spec.json';assert not path.exists();save(path,spec);run('scripts/evaluate_decision_policy_iteration.py','--spec',str(path))
 phase('complete',expanded=True,next='Audit refreshed pool and validation64; freeze final test or one bounded adjustment based only on validation. No final test launched.')
if __name__=='__main__':
 try:main()
 except BaseException as e:phase('failed',error=repr(e));traceback.print_exc();raise
