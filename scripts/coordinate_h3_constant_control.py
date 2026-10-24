"""Token-only train calibration, freeze one control, then paired dev evaluation."""
import json,os,subprocess,time
from pathlib import Path
from run_decision_policy_pilot import ROOT,REPO,save,sha
NAME='decision_h3_constant_control_20261024'
CAL='decision_h3_constant_calibration24_20261024'
DEV='decision_h3_constant_validation64_20261024'
def read(p):return json.loads(p.read_text())
def main():
 reg=REPO/'lab/registry'/NAME;plan=read(reg/'plan.json');state=ROOT/'results/decision_h3_constant_coordinator_20261024.json';assert not state.exists()
 for p,h in plan['input_sha256'].items():assert sha(Path(p))==h
 for p,h in plan['source_sha256'].items():assert sha(REPO/p)==h
 start=time.monotonic()
 def phase(name,**kw):
  data=dict(phase=name,pid=None if name in ['complete','failed'] else os.getpid(),seconds=time.monotonic()-start,**kw);save(state,data)
  m=read(REPO/'memory/EXPERIMENT_STATUS.json');m['h3_constant_control']=data;save(REPO/'memory/EXPERIMENT_STATUS.json',m)
 def run(script,args,timeout):
  log=ROOT/'logs'/f'h3_constant_{Path(script).stem}_{time.time_ns()}.log'
  with log.open('x') as f:
   child=subprocess.Popen([str(ROOT/'runtime/bin/python'),script,*args],cwd=REPO,env=dict(os.environ,PYTHONPATH='src:scripts',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
   phase('running',child_pid=child.pid,script=script,args=args,log=str(log))
   try:rc=child.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    child.terminate()
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:child.kill();child.wait()
    raise
   if rc:raise RuntimeError(f'{script} exited{rc}: {log}')
 try:
  run('scripts/reporting/archive_decision_run.py',['--run',NAME],300)
  run('scripts/evaluate_decision_stop_value_h2.py',['--spec',str(reg/'calibration_spec.json')],2100)
  p=ROOT/'results'/CAL;status=read(p/'status.json');assert status['phase']=='complete'
  spec=read(p/'spec.json');eps=[read(f) for f in (p/'episodes').glob('*.json')]
  assert len(eps)==len(spec['qids'])*len(spec['policies'])==status['episodes']
  for k,f in [('generated','generated_tokens'),('inputs','prompt_tokens')]:assert sum(e[f] for e in eps)==status[k]
  # Selection uses raw generated tokens only. Answer scores are never consulted.
  means={name:sum(read(p/'episodes'/f'{name}_{q}.json')['generated_tokens'] for q in spec['qids'])/len(spec['qids']) for name in spec['policies']}
  best=min(plan['candidates'],key=lambda c:(abs(means[c['name']]-means['h3_reference']),abs(c['bias']),c['name']))
  ratio=means[best['name']]/means['h3_reference'];selection=dict(selected=best,generated_means=means,cost_ratio=ratio,cost_match_pass=abs(ratio-1)<=plan['cost_match_tolerance'],f1_used=False,checkpoint=spec['policies'][best['name']])
  save(reg/'selection.json',selection)
  run('scripts/reporting/archive_decision_run.py',['--run',CAL],300)
  dev=read(REPO/'lab/registry/decision_budget_stop_h3_20261024/validation_spec.json')
  assert dev['qids']==plan['validation'] and not set(dev['qids'])&set(spec['qids'])
  dev.update(run_name=DEV,selection='H3 conditional residual vs token-calibrated constant, selected only on train; no dev retuning',policies={'selected_reference':spec['policies']['h3_reference'],'constant_stop':selection['checkpoint']})
  dev['source_sha256']={p:sha(REPO/p) for p in dev['source_sha256']};save(reg/'validation_spec.json',dev)
  run('scripts/evaluate_decision_stop_value_h2.py',['--spec',str(reg/'validation_spec.json')],2100)
  run('scripts/reporting/summarize_decision_shared_evaluation.py',['--run',DEV,'--baseline','selected_reference'],180)
  run('scripts/reporting/compare_decision_policy_campaign.py',['--run',DEV],180)
  run('scripts/reporting/archive_decision_run.py',['--run',DEV],300)
  phase('complete',report=f'lab/reports/{DEV}/report.md',selection=selection,goal_achieved=False)
 except BaseException as error:phase('failed',error=repr(error));raise
if __name__=='__main__':main()
