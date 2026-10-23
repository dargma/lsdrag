"""Bounded sequential coordinator: collector -> CPU heads -> validation -> cost sweep.
No final test access here. Stops on worker failure; no blind retry.
"""
from pathlib import Path
import json,os,subprocess,time,traceback
from datetime import datetime,timezone
REPO=Path(__file__).resolve().parents[1];ROOT=Path('/content/local_fast/lsdrag-text-v1');PY=ROOT/'runtime/bin/python';DATA=ROOT/'results/decision_state_outcomes_20261023';HEADS=ROOT/'results/decision_state_heads_20261023';STATE=ROOT/'results/decision_state_coordinator_20261023.json'
def read(p):return json.loads(p.read_text())
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def phase(p,**kw):save(STATE,dict(phase=p,pid=None if p in ['complete','failed'] else os.getpid(),**kw))
def remaining():return datetime(2026,10,23,21,48,tzinfo=timezone.utc).timestamp()-time.time()
def run(script,*args):
 if remaining()<600:raise TimeoutError('Insufficient time before reserved reporting hour')
 log=ROOT/'logs'/('state_stage_'+Path(script).stem+'_'+str(int(time.time()))+'.log')
 phase('running',script=script,log=str(log))
 with log.open('w') as f:
  p=subprocess.Popen([str(PY),script,*args],cwd=REPO,env=dict(os.environ,PYTHONPATH='src'),stdout=f,stderr=subprocess.STDOUT)
  phase('running',script=script,child_pid=p.pid,log=str(log));
  try:p.wait(timeout=remaining())
  except subprocess.TimeoutExpired:
   p.terminate();p.wait(timeout=30);raise
  if p.returncode:raise RuntimeError(f'{script} failed code{p.returncode}; {log}')
def evaluate(names,label):
 m=read(DATA/'manifest.json');h=read(HEADS/'manifest.json');spec={'split':'validation','run_name':label,'candidates':names,'qids':m['validation'],'head_sha256':{n:h['candidates'][n]['sha256'] for n in names},'max_seconds':1800,'max_generated':600000,'max_inputs':16000000}
 path=REPO/'lab/registry/decision_state_outcomes_20261023'/f'{label}_spec.json';assert not path.exists();save(path,spec);run('scripts/evaluate_decision_state_heads.py','--spec',str(path));return read(ROOT/'results'/label/'summary.json')['policies']
def main():
 phase('waiting_collection')
 while True:
  s=read(DATA/'status.json')
  if s['phase']=='complete':break
  if s['phase']=='failed':raise RuntimeError('Collector failed; preserve artifacts')
  p=Path(f'/proc/{s.get("pid")}/cmdline')
  if not p.exists() or b'run_decision_state_outcomes.py' not in p.read_bytes():raise RuntimeError('Collector disappeared')
  if remaining()<600:raise TimeoutError('Deadline')
  time.sleep(15)
 run('scripts/fit_decision_state_heads.py');h=read(HEADS/'manifest.json');results=evaluate(list(h['candidates']),'decision_state_validation1_20261023');base=results['grpo_baseline']
 names=['same_structure_improved','last_linear','last_mlp','multi_linear','multi_mlp']
 def rank(n):
  v=results[n];quality=(v['f1']-base['f1'])/.03;tokens=(1-v['generated_tokens']/base['generated_tokens'])/.15
  return (min(quality,tokens),v['utility'],n)
 best=max(names,key=rank)
 save(HEADS/'validation_stage1_selection.json',{'selected_for_cost_sweep':best,'criterion':'max min(F1_delta/.03,generated_reduction/.15), then utility; validation only','results':results})
 # Dynamic choice: sweep only when at least one primary component improves.
 if results[best]['f1']>base['f1'] or results[best]['generated_tokens']<base['generated_tokens']:
  run('scripts/fit_decision_state_heads.py','--cost-sweep',best)
  swept=[best+'_lambda0.01',best+'_lambda0.06'];results.update(evaluate(swept,'decision_state_validation2_20261023'));names+=swept
 selected=sorted(names,key=rank,reverse=True)[:2]
 save(HEADS/'validation_selection.json',{'selected':selected,'all_validation':results,'criterion':'max min(F1_delta/.03,generated_reduction/.15), then utility; freeze before test','test_not_evaluated':True})
 run('scripts/run_decision_grpo_continuation.py')
 phase('complete',selected=selected,next='Inspect validation and ordinary GRPO control; freeze final test or bounded train-only refinement. No test launched by this coordinator.')
if __name__=='__main__':
 try:main()
 except BaseException as e:phase('failed',error=repr(e));traceback.print_exc();raise
