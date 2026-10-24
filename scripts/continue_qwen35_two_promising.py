"""One finite two-candidate evaluation, verification and archival; no retry."""
import os,json,subprocess,time,sys
from pathlib import Path
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[1]
NAME='qwen35_9b_two_promising_20261024';RUN='qwen35_9b_two_promising_validation64_20261024'
statuspath=ROOT/'results'/NAME/'coordinator_status.json'
def status(phase,**kw):statuspath.write_text(json.dumps(dict(phase=phase,pid=os.getpid() if phase=='running' else None,updated_epoch=time.time(),**kw),indent=2)+'\n')
def main():
 env=os.environ.copy();env['PYTHONPATH']='src:scripts'
 commands=[('archive_training',['scripts/reporting/archive_decision_run.py','--run',NAME]),('evaluation',['scripts/evaluate_qwen35_h3.py','--spec',str(REPO/'lab/registry'/NAME/'evaluation_spec.json')]),('verification',['scripts/reporting/verify_qwen35_two_promising.py']),('archive_evaluation',['scripts/reporting/archive_decision_run.py','--run',RUN])]
 try:
  for stage,args in commands:
   if stage=='evaluation':
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]);assert free>=40960,f'Insufficient VRAM: {free}MiB'
   log=ROOT/'logs'/f'{NAME}_{stage}.log'
   with log.open('x') as f:
    child=subprocess.Popen([sys.executable,'-u']+args,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
    status('running',stage=stage,child_pid=child.pid,log=str(log));code=child.wait();assert code==0,f'{stage} exited {code}; inspect {log}'
  status('complete',note='Raw evaluation/audit/archives complete. Review conclusions and persist final memory/Git/code backup.')
 except BaseException as e:status('failed',error=repr(e));raise
if __name__=='__main__':main()
