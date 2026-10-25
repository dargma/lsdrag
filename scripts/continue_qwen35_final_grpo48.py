"""Finite final train/audit/archive/paired-eval/audit/archive; no retries or extensions."""
import os,json,subprocess,time,sys,hashlib,tarfile,shutil
from pathlib import Path
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[1]
NAME='qwen35_final_grpo48_20261025';EVAL='qwen35_final_grpo_validation64_20261025';REG=REPO/'lab/registry'/NAME
STATUS=ROOT/'results'/f'{NAME}_pipeline';STATUS.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def status(phase,**kw):save(STATUS/'status.json',dict(phase=phase,pid=os.getpid() if phase=='running' else None,updated_epoch=time.time(),**kw))
def archive(name):
 run=ROOT/'results'/name;assert read(run/'status.json')['phase']=='complete'
 local=ROOT/f'{name}.tar.gz'
 with tarfile.open(local,'w:gz') as t:t.add(run,arcname=name)
 dest=Path('/content/drive/MyDrive/lsdrag-backups/20261025')/local.name;assert not dest.exists();shutil.copyfile(local,dest);h=sha(local);assert sha(dest)==h
 save(REPO/'lab/registry'/name/'archive.json',dict(path=str(dest),sha256=h,bytes=dest.stat().st_size,run=str(run),phase='complete',copy_verified=True,backbone_weights_included=False))
def command(stage,args):
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
 if stage in ['training','evaluation']:assert free>=40960,f'Insufficient GPU free memory: {free}'
 log=ROOT/'logs'/f'{NAME}_{stage}.log';env=os.environ.copy();env['PYTHONPATH']='src:scripts'
 with log.open('x') as f:
  p=subprocess.Popen([sys.executable,'-u']+args,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT);status('running',stage=stage,child_pid=p.pid,log=str(log));code=p.wait();assert code==0,f'{stage} failed {code}; see {log}'
def main():
 try:
  frozen=read(REG/'pipeline_sources.json')
  for p,h in frozen.items():assert sha(REPO/p)==h,p
  command('training',['scripts/run_qwen35_final_grpo48.py'])
  command('training_audit',['scripts/reporting/audit_qwen35_final_grpo48.py']);archive(NAME)
  spec=read(REG/'evaluation_template.json');cp=ROOT/'results'/NAME/'head.pt';spec['policies']['after']=dict(kind='qwen35_base',checkpoint=str(cp),sha256=sha(cp));dest=REG/'evaluation_spec.json';assert not dest.exists();save(dest,spec)
  for p,h in frozen.items():assert sha(REPO/p)==h,p
  command('evaluation',['scripts/evaluate_qwen35_h3.py','--spec',str(dest)])
  command('evaluation_audit',['scripts/reporting/audit_qwen35_final_grpo_eval.py']);archive(EVAL)
  status('complete',note='Final bounded experiment complete; human synthesis and final memory/Git/code backup remain. No subsequent experiments.')
 except BaseException as e:status('failed',error=repr(e));raise
if __name__=='__main__':main()
