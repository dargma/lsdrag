"""One-shot completion watcher: audit, archive, memory and scoped Git save."""
from pathlib import Path
import json,os,time,subprocess,tarfile,hashlib,traceback
REPO=Path(__file__).resolve().parents[1];ROOT=Path('/content/local_fast/lsdrag-text-v1');RUN=ROOT/'results/decision_ppo96_20261023';PY=ROOT/'runtime/bin/python';STATE=RUN/'finalizer_status.json'
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 deadline=time.monotonic()+5400;write(STATE,{'phase':'waiting','pid':os.getpid(),'run':str(RUN)})
 while time.monotonic()<deadline:
  s=json.loads((RUN/'status.json').read_text())
  if s['phase']=='complete':break
  if s['phase']=='failed':raise RuntimeError('Worker failed: '+str(s))
  pid=s.get('pid');cmd=Path(f'/proc/{pid}/cmdline')
  if not pid or not cmd.exists() or b'run_decision_ppo96.py' not in cmd.read_bytes():raise RuntimeError('Worker missing; partial artifacts preserved, no auto restart')
  time.sleep(15)
 else:raise TimeoutError('Completion watcher budget exceeded')
 write(STATE,{'phase':'auditing','pid':os.getpid()})
 subprocess.run([str(PY),'scripts/reporting/summarize_decision_ppo96.py'],cwd=REPO,check=True)
 result=json.loads((REPO/'lab/reports/decision_ppo96_20261023/result.json').read_text());archive=Path('/content/drive/MyDrive/lsdrag-backups/20261023/decision_ppo96_20261023.tar.gz');archive.parent.mkdir(parents=True,exist_ok=True)
 # Exclude this watcher's mutable status; retain all worker outputs and head checkpoints.
 with tarfile.open(archive,'w:gz') as t:
  for p in sorted(RUN.iterdir()):
   if p.name!='finalizer_status.json':t.add(p,arcname=RUN.name+'/'+p.name)
 with tarfile.open(archive) as t:assert any(p.name.endswith('summary.json') for p in t.getmembers())
 record=REPO/'lab/registry/decision_ppo96_20261023/archive.json';write(record,{'path':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size})
 v=result['policies'];comparison=result['ppo_minus']['tempered_anchor'];entry=f'''## 2026-10-23 — PPO96 자동 완료 검증·저장\n\nrun {RUN};source ced20cc,train96×4×3=1152+fresh40×4=160,total1312. worker complete. Raw/headhash/probability audit통과. PPO F1={v['ppo']['f1']:.6f},anchor={v['tempered_anchor']['f1']:.6f},RWR={v['rwr']['f1']:.6f}. PPO−anchor={comparison}. 사전gate={result['strong_signal_gate_passed']}. 실제optimizersteps각learner={result['optimizer_steps_per_learner']}. 보고 docs/JEV_DECISION_PPO96_REPORT_20261023.md. rawDrivearchiveSHA검증 registry/decision_ppo96_20261023/archive.json. 독립seed1,새40노출보호. 이전24와질문/수집/update/평가셋이달라순수규모효과주장금지. 후속새학습자동기동없음.\n\n'''
 for name in ['CURRENT.md','HEAD_EXPERIMENT.md']:
  p=REPO/'memory'/name;p.write_text(entry+p.read_text())
 with (REPO/'memory/SESSION_LOG.md').open('a') as f:f.write('\n'+entry.rstrip()+'\n')
 p=REPO/'memory/EXPERIMENT_STATUS.json';s=json.loads(p.read_text());s['decision_ppo96'].update(phase='complete',pid=None,session_id=None,finalizer_pid=None,report='docs/JEV_DECISION_PPO96_REPORT_20261023.md',gate_passed=result['strong_signal_gate_passed']);write(p,s)
 paths=['docs/JEV_DECISION_PPO96_REPORT_20261023.md','lab/reports/decision_ppo96_20261023/result.json','lab/registry/decision_ppo96_20261023/archive.json','memory/CURRENT.md','memory/HEAD_EXPERIMENT.md','memory/SESSION_LOG.md','memory/EXPERIMENT_STATUS.json']
 branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
 if branch!='jev_v1.2':raise RuntimeError('Report saved; branch changed, Git pending')
 subprocess.run(['git','add','--',*paths],cwd=REPO,check=True)
 subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','--only','-m','report: complete PPO96 expanded trajectory experiment','--',*paths],cwd=REPO,check=True)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();subprocess.run(['git','push','origin','jev_v1.2'],cwd=REPO,check=True)
 write(STATE,{'phase':'complete','pid':None,'commit':commit,'report':'docs/JEV_DECISION_PPO96_REPORT_20261023.md'})
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(STATE,{'phase':'failed','pid':None,'error':repr(e),'note':'Artifacts preserved; inspect log before retry'});traceback.print_exc();raise
