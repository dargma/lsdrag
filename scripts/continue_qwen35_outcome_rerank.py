"""One outcome-readout hypothesis; wait for verified fresh data and prior GPU completion."""
import os,json,time,subprocess,sys
from pathlib import Path
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[1];NAME='qwen35_outcome_rerank_20261025';EVAL='qwen35_outcome_rerank_validation64_20261025';REG=REPO/'lab/registry'/NAME
# Reuse only CPU archive/hash helpers; importing does not start another experiment.
from continue_qwen35_final_grpo48 import archive,sha,save
STATUS=ROOT/'results'/f'{NAME}_pipeline';STATUS.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def status(phase,**kw):save(STATUS/'status.json',dict(phase=phase,pid=os.getpid() if phase in ['waiting','running'] else None,updated_epoch=time.time(),**kw))
def command(stage,args):
 if stage=='evaluation':assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])>=40960
 log=ROOT/'logs'/f'{NAME}_{stage}.log';env=os.environ.copy();env['PYTHONPATH']='src:scripts'
 with log.open('x') as f:
  p=subprocess.Popen([sys.executable,'-u']+args,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT);status('running',stage=stage,child_pid=p.pid,log=str(log));assert p.wait()==0,f'{stage} failed; inspect {log}'
def main():
 try:
  deadline=time.monotonic()+12000
  previous=ROOT/'results/qwen35_final_grpo48_20261025_pipeline/status.json';train=ROOT/'results/qwen35_final_grpo48_20261025/status.json';audit=REPO/'lab/reports/qwen35_final_grpo48_20261025/audit.json'
  status('waiting',stage='verified_training_data')
  while True:
   p=read(previous);assert p['phase']!='failed',p
   if train.exists() and read(train)['phase']=='complete' and audit.exists():break
   assert time.monotonic()<deadline,'Predecessor wait timed out; no automatic restart';time.sleep(10)
  plan=read(REG/'plan.json')
  for f,h in plan['source_sha256'].items():assert sha(REPO/f)==h
  command('fit',['scripts/fit_qwen35_outcome_rerank.py']);archive(NAME)
  fit=read(ROOT/'results'/NAME/'summary.json')
  if not fit['predictive_gate_passed']:
   status('complete',predictive_gate_passed=False,evaluation_launched=False,note='Predictive feasibility failed; preserve negative result, no retuning or GPU evaluation.');return
  status('waiting',stage='prior_gpu_experiment_complete')
  while read(previous)['phase']!='complete':
   assert read(previous)['phase']!='failed';assert time.monotonic()<deadline;time.sleep(10)
  spec=read(REG/'evaluation_template.json');spec['policies']['after']=dict(kind='outcome_rerank',checkpoint=fit['checkpoint'],sha256=fit['checkpoint_sha256']);dest=REG/'evaluation_spec.json';assert not dest.exists();save(dest,spec)
  command('evaluation',['scripts/evaluate_qwen35_outcome_rerank.py','--spec',str(dest)]);command('evaluation_audit',['scripts/reporting/audit_qwen35_outcome_rerank_eval.py']);archive(EVAL)
  status('complete',predictive_gate_passed=True,evaluation_launched=True,note='One candidate completed; final interpretation/memory/Git/code backup remain.')
 except BaseException as e:status('failed',error=repr(e));raise
if __name__=='__main__':main()
