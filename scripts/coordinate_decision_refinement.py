"""Wait ordinary GRPO control; validate three refinements; replicate selected primary."""
from pathlib import Path
import json,os,time,subprocess,traceback
from datetime import datetime,timezone
REPO=Path(__file__).resolve().parents[1];ROOT=Path('/content/local_fast/lsdrag-text-v1');PY=ROOT/'runtime/bin/python';HEADS=ROOT/'results/decision_state_heads_20261023';DATA=ROOT/'results/decision_state_outcomes_20261023';STATE=ROOT/'results/decision_refinement_coordinator_20261023.json'
def read(p):return json.loads(p.read_text())
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def phase(name,**kw):save(STATE,dict(phase=name,pid=None if name in ['complete','failed'] else os.getpid(),**kw))
def remain():return datetime(2026,10,23,21,48,tzinfo=timezone.utc).timestamp()-time.time()
def run(script,*args):
 if remain()<600:raise TimeoutError('Reporting-hour reserve')
 log=ROOT/'logs'/f'refinement_{Path(script).stem}_{int(time.time())}.log'
 with log.open('w') as f:
  p=subprocess.Popen([str(PY),script,*args],cwd=REPO,env=dict(os.environ,PYTHONPATH='src'),stdout=f,stderr=subprocess.STDOUT)
  phase('running',script=script,child_pid=p.pid,log=str(log))
  try:p.wait(timeout=remain())
  except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30);raise
  if p.returncode:raise RuntimeError(f'Failed {script}; inspect {log}')
def evaluate(names,label):
 m=read(DATA/'manifest.json');h=read(HEADS/'manifest.json');spec={'split':'validation','run_name':label,'candidates':names,'qids':m['validation'],'head_sha256':{n:h['candidates'][n]['sha256'] for n in names},'max_seconds':1800,'max_generated':600000,'max_inputs':16000000}
 path=REPO/'lab/registry/decision_state_outcomes_20261023'/f'{label}_spec.json';assert not path.exists();save(path,spec);run('scripts/evaluate_decision_state_heads.py','--spec',str(path));return read(ROOT/'results'/label/'summary.json')['policies']
def main():
 phase('waiting_previous_coordinator')
 while True:
  s=read(ROOT/'results/decision_state_coordinator_20261023.json')
  if s['phase']=='complete':break
  if s['phase']=='failed':raise RuntimeError('Previous coordinator failed')
  p=Path(f'/proc/{s.get("pid")}/cmdline')
  if not p.exists() or b'coordinate_decision_state_stages.py' not in p.read_bytes():raise RuntimeError('Previous coordinator disappeared')
  if remain()<600:raise TimeoutError('Deadline')
  time.sleep(15)
 names=['multi_mlp_cost006','multi_mlp_generation015','last_linear_cost009'];new=evaluate(names,'decision_state_validation3_20261023');old=read(HEADS/'validation_selection.json');results=old['all_validation']|new;h=read(HEADS/'manifest.json');base=results['grpo_baseline']
 pool=[n for n in results if n!='grpo_baseline' and not n.endswith('warmstart')]
 def rank(n):
  v=results[n];return (min((v['f1']-base['f1'])/.05,(1-v['generated_tokens']/base['generated_tokens'])/.30),v['utility'],n)
 primary=max(pool,key=rank)
 def structure(n):
  d=h['candidates'][n];return (d.get('inputs','last'),d.get('body','original_mlp_skip'))
 other=[n for n in pool if structure(n)!=structure(primary)];secondary=max(other,key=rank) if other else None
 save(HEADS/'final_candidate_selection_before_seed2.json',{'primary':primary,'secondary':secondary,'target':{'f1_delta':.05,'generated_ratio':.7},'criterion':'max min(F1delta/.05,generated_reduction/.30), then utility; secondary best distinct structure','all_validation':results,'test_evaluated':False,'seed2_precommitted':True})
 run('scripts/run_decision_state_outcomes_seed2.py');run('scripts/fit_decision_selected_seed2.py');evaluate([primary+'_seed2','grpo_baseline'],'decision_state_validation_seed2_20261023')
 phase('complete',primary=primary,secondary=secondary,next='Audit seed2; freeze final evaluation including both primary seeds and matched controls. No final test launched.')
if __name__=='__main__':
 try:main()
 except BaseException as e:phase('failed',error=repr(e));traceback.print_exc();raise
