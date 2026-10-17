"""One-shot launch after a fresh, continuously observed 15-minute idle window.

Uses local telemetry only; no old Codex session or unrelated process is resumed.
"""
import fcntl,json,os,subprocess,time,tarfile,hashlib,shutil
from pathlib import Path
from watch_gpu_schedule import sample

REPO=Path(__file__).resolve().parents[1]
REC=Path('/content/local_fast/lsdrag-recovery/20261017')
OUT=Path('/content/local_fast/lsdrag-text-v1/results/improvement_v2_stage1_20261017')
PYTHON='/content/local_fast/lsdrag-text-v1/experiment-env/bin/python'

def save(phase,**extra):
    value={'phase':phase,'pid':os.getpid(),'epoch':time.time(),**extra}
    tmp=REC/'launch_status.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(REC/'launch_status.json')


def main():
    lock=(REC/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (REC/'launch_intent.json').exists():raise RuntimeError('One-shot already claimed')
    save('waiting_for_900_seconds_idle')
    while not (REC/'STOP').exists():
        try:
            state=json.loads((REC/'idle_status.json').read_text())
            eligible=state.get('eligible') and 0<=time.time()-state['epoch']<=65
            if eligible and sample('GPU-619f1d71-c0b4-21cf-abcd-4a50a8549f3f')['idle']:
                command=[PYTHON,'-u','scripts/diagnose_text_generation_paths.py','--output',str(OUT),'--minutes','15']
                intent={'epoch':time.time(),'idle_evidence':state,'command':command}
                with (REC/'launch_intent.json').open('x') as f:json.dump(intent,f,indent=2)
                with (REC/'stage1.log').open('a') as log:
                    env=os.environ.copy();env['PYTHONPATH']=str(REPO/'src')
                    child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    save('stage1_running',child_pid=child.pid,output=str(OUT));code=child.wait()
                save('stage1_finished' if code==0 else 'stage1_failed',exit_code=code,output=str(OUT))
                summary=json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {'error':'See recovery stage1.log'}
                report=REPO/'lab/reports/text_integrated_20261016/improvement_v2_stage1_20261017.json'
                report.write_text(json.dumps({'exit_code':code,'output':str(OUT),'summary':summary,'next':'Review stage1 evidence before stage2. No learned-policy improvement claim.'},indent=2)+'\n')
                head='## 2026-10-17 — 개선v2 stage1 '+('완료' if code==0 else '실패')+'\n\n'+json.dumps(summary,ensure_ascii=False)+'\n\n다음: token/cache/schema 증거 검토 후stage2. stage2/3/4 미실행.\n\n'
                for name in ['CURRENT.md','HEAD_EXPERIMENT.md']:
                    p=REPO/'memory'/name;p.write_text(head+p.read_text())
                p=REPO/'memory/SESSION_LOG.md';p.write_text(p.read_text()+'\n'+head)
                p=REPO/'memory/EXPERIMENT_STATUS.json';status=json.loads(p.read_text());status.update(phase='improvement_v2_stage1_complete' if code==0 else 'improvement_v2_stage1_failed',pid=None,next='Review stage1 evidence before stage2',improvement_stage1=summary);p.write_text(json.dumps(status,indent=2)+'\n')
                archive=REC/'improvement_v2_stage1_20261017.tar.gz'
                with tarfile.open(archive,'w:gz') as tar:
                    if OUT.exists():tar.add(OUT,arcname=OUT.name)
                    tar.add(REC/'stage1.log',arcname='stage1.log');tar.add(REC/'launch_intent.json',arcname='launch_intent.json')
                dest=Path('/content/drive/MyDrive/lsdrag-backups/20261017');dest.mkdir(parents=True,exist_ok=True);shutil.copyfile(archive,dest/archive.name)
                sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
                assert sha(archive)==sha(dest/archive.name)
                (REPO/'lab/registry/recovery_20261017/stage1_archive.json').write_text(json.dumps({'archive':str(dest/archive.name),'sha256':sha(archive)},indent=2)+'\n')
                files=['memory/CURRENT.md','memory/HEAD_EXPERIMENT.md','memory/SESSION_LOG.md','memory/EXPERIMENT_STATUS.json',str(report.relative_to(REPO)),'lab/registry/recovery_20261017/stage1_archive.json']
                subprocess.run(['git','add','--',*files],cwd=REPO,check=True)
                subprocess.run(['git','commit','-m','Record restored improvement v2 stage1 evidence'],cwd=REPO,check=True)
                pushed=subprocess.run(['git','push','origin','jev_v1.1'],cwd=REPO).returncode==0
                save('stage1_recorded',exit_code=code,git_pushed=pushed,summary=str(report));return
        except (OSError,ValueError,subprocess.SubprocessError) as e:
            save('launcher_error',error=str(e))
            if (REC/'launch_intent.json').exists():raise
        time.sleep(10)
    save('cancelled_before_launch')

if __name__=='__main__':main()
