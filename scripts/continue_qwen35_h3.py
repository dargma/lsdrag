"""Finite sequential9B experiment; stop on any failed gate, never retry GPU jobs."""
import argparse,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[1];REG=REPO/'lab/registry/qwen35_9b_h3_20261024';OUT=ROOT/'results/qwen35_9b_pipeline_20261024'
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int,required=True);args=ap.parse_args();OUT.mkdir(exist_ok=True)
 lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def status(phase,**kw):
  p=OUT/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(phase=phase,pid=None if phase in ['complete','failed'] else os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw),indent=2)+'\n');tmp.replace(p)
 def frozen():
  for f,h in read(REG/'pipeline_sources.json').items():assert hashlib.sha256((REPO/f).read_bytes()).hexdigest()==h,f
 def command(stage,argv,gpu=False):
  frozen()
  if gpu:
   while True:
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip());
    if free>=40960:break
    status('waiting_memory',stage=stage,free_mib=free);time.sleep(15)
  log=ROOT/'logs'/f'qwen35_pipeline_{stage}_20261024.log';env=os.environ.copy();env['PYTHONPATH']='src:scripts'
  with log.open('x') as f:
   child=subprocess.Popen([str(ROOT/'runtime/bin/python'),'-u',*argv],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
   status('running',stage=stage,child_pid=child.pid,log=str(log));code=child.wait()
  assert code==0,f'{stage} exited{code}; see{log}'
 def audit(name):command('audit_'+name,['scripts/reporting/verify_qwen35_training.py','--run',name])
 def archive(name):command('archive_'+name,['scripts/reporting/archive_decision_run.py','--run',name])
 try:
  frozen()
  while Path(f'/proc/{args.wait_pid}').exists():
   cmd=Path(f'/proc/{args.wait_pid}/cmdline').read_bytes()
   if not cmd:break
   assert b'collect_qwen35_shared_teacher.py' in cmd,'Wait PID identity changed'
   status('waiting_teacher',child_pid=args.wait_pid);time.sleep(15)
  assert read(ROOT/'results/qwen35_9b_teacher192_20261024/status.json')['phase']=='complete','Teacher incomplete; do not resume silently'
  audit('qwen35_9b_teacher192_20261024');archive('qwen35_9b_teacher192_20261024')
  command('bootstrap',['scripts/fit_qwen35_bootstrap.py']);archive('qwen35_9b_bootstrap_20261024')
  command('prepare_outcomes',['scripts/prepare_qwen35_stage.py','--stage','outcomes'])
  command('outcomes',['scripts/collect_qwen35_outcomes.py','--spec',str(REG/'outcomes_spec.json')],gpu=True)
  audit('qwen35_9b_outcomes48_20261024');archive('qwen35_9b_outcomes48_20261024')
  command('fit_base',['scripts/fit_qwen35_outcome_heads.py','--stage','base']);archive('qwen35_9b_base_20261024')
  command('prepare_stop',['scripts/prepare_qwen35_stage.py','--stage','stop'])
  command('stop',['scripts/collect_qwen35_outcomes.py','--spec',str(REG/'stop_spec.json')],gpu=True)
  audit('qwen35_9b_stop48_20261024');archive('qwen35_9b_stop48_20261024')
  command('fit_stop',['scripts/fit_qwen35_outcome_heads.py','--stage','stop']);archive('qwen35_9b_h3_20261024')
  command('prepare_evaluation',['scripts/prepare_qwen35_stage.py','--stage','evaluation'])
  command('evaluation',['scripts/evaluate_qwen35_h3.py','--spec',str(REG/'evaluation_spec.json')],gpu=True)
  command('audit_evaluation',['scripts/reporting/verify_qwen35_eval.py']);archive('qwen35_9b_h3_validation64_20261024')
  status('complete',note='Raw experiment stages complete; human-facing synthesis, cross-model comparison, final provenance push and completion audit still required.')
 except BaseException as e:status('failed',error=repr(e));raise
if __name__=='__main__':main()
