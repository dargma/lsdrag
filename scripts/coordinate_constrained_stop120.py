"""Finish the existing collection, audit, fixed fit and gated train120 only."""
import datetime,json,os,subprocess,time,hashlib
from pathlib import Path
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path('/content/local_fast/lsdrag-code')
COLLECTION='decision_h3_stop_coverage48_20261024';FIT='decision_constrained_stop120_20261024';EVAL='decision_constrained_stop120_train_eval_20261024'
STATE=ROOT/'results/decision_constrained_stop120_pipeline_20261024.json'
def read(p):return json.loads(p.read_text())
def save(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n')
def phase(name,**extra):
 state=dict(phase=name,pid=None if name in ['complete','failed','fit_gate_failed','train_gate_failed'] else os.getpid(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra);save(STATE,state)
 p=REPO/'memory/EXPERIMENT_STATUS.json';m=read(p);m['coverage120_pipeline']=state;save(p,m)
 entry='## '+state['updated_utc']+' — coverage120 pipeline '+name+'\n\n'+json.dumps(state,ensure_ascii=False)+'\n\nGoal ACTIVE; reserved80 protected; no automatic development launch.\n\n'
 for name in ['CURRENT.md','HEAD_EXPERIMENT.md']:
  p=REPO/'memory'/name;p.write_text(entry+p.read_text())
 p=REPO/'memory/SESSION_LOG.md';p.write_text(p.read_text()+'\n\n'+entry)
def run(script,args=(),timeout=600):
 log=ROOT/'logs'/f'coverage120_{Path(script).stem}_{int(time.time())}.log'
 with log.open('x') as f:
  child=subprocess.Popen([str(ROOT/'runtime/bin/python'),script,*args],cwd=REPO,env=dict(os.environ,PYTHONPATH='src:scripts',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
  phase('running',script=script,child_pid=child.pid,log=str(log))
  try:rc=child.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   child.terminate()
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:child.kill();child.wait()
   raise
  assert rc==0,(script,rc,str(log))
def main():
 assert not STATE.exists(),'Do not duplicate a pipeline'
 try:
  collection=ROOT/'results'/COLLECTION
  status=read(collection/'status.json');pid=status.get('pid')
  if status['phase'] not in ['complete','failed']:
   assert pid and b'run_decision_h3_stop_coverage48.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
   phase('waiting_collection',collection_pid=pid)
   until=time.monotonic()+2000
   while True:
    status=read(collection/'status.json')
    if status['phase'] in ['complete','failed']:break
    assert time.monotonic()<until,'Collection wait budget exceeded'
    # A disappearing handle is not restart authorization. Re-read terminal status.
    if not Path(f'/proc/{pid}').exists():
     time.sleep(1);status=read(collection/'status.json');assert status['phase'] in ['complete','failed'];break
    assert b'run_decision_h3_stop_coverage48.py' in Path(f'/proc/{pid}/cmdline').read_bytes(),'PID identity changed'
    time.sleep(10)
  assert status['phase']=='complete',status
  run('scripts/reporting/archive_decision_run.py',['--run',COLLECTION])
  run('scripts/reporting/verify_h3_stop_coverage48.py')
  run('scripts/prepare_decision_constrained_stop120.py')
  run('scripts/fit_decision_constrained_stop120.py')
  run('scripts/reporting/archive_decision_run.py',['--run',FIT])
  result=read(ROOT/'results'/FIT/'summary.json')
  if not result['learning_progress']:phase('fit_gate_failed',summary=str(ROOT/'results'/FIT/'summary.json'));return
  spec=read(REPO/'lab/registry/decision_constrained_stop72_20261024/train_spec.json');plan=read(REPO/'lab/registry'/FIT/'plan.json')
  spec.update(run_name=EVAL,split='training120',qids=plan['fit'],policies={'h3':plan['reference_spec'],'constrained_stop':result['candidate']})
  # Existing native evaluator sources must remain identical, not silently rehashed.
  for p,h in spec['source_sha256'].items():assert hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h
  specpath=REPO/'lab/registry'/FIT/'train_spec.json';assert not specpath.exists();save(specpath,spec)
  run('scripts/evaluate_decision_stop_value_h2.py',['--spec',str(specpath)],1500)
  run('scripts/reporting/verify_constrained_stop120_train.py')
  run('scripts/reporting/archive_decision_run.py',['--run',EVAL])
  report=read(REPO/'lab/reports'/EVAL/'verification.json');phase('complete' if report['train_gate'] else 'train_gate_failed',train_gate=report['train_gate'],report=str(REPO/'lab/reports'/EVAL/'verification.json'))
 except BaseException as e:
  phase('failed',error=repr(e));raise
if __name__=='__main__':main()
