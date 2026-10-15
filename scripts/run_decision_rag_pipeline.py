"""Recoverable serial pipeline: finish budget diagnosis, shared1024 stage1, stage2."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from checkpoint_decision_rag import checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait-pid',type=int);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];root=Path('/content/local_fast/lsdrag-head-v13');reports=repo/'lab/reports/decision_rag_20261015'
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(phase,**fields):
        row={'phase':phase,'pid':os.getpid(),'updated_epoch':time.time(),**fields}
        tmp=root/'pipeline_status.tmp';tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(root/'pipeline_status.json');print(json.dumps(row),flush=True)
    def command(cmd,logname):
        log=root/'logs'/logname
        with log.open('w') as stream:
            proc=subprocess.Popen(cmd,cwd=repo,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':'src'})
            status('running',child_pid=proc.pid,command=cmd,log=str(log));code=proc.wait()
        if code:raise RuntimeError(f'Command failed{code}: {log}')
    def diagnose_report(run,destination):
        subprocess.run([sys.executable,'scripts/report_decision_rag.py','--run',str(run),'--report',str(destination),'--limit','20'],cwd=repo,check=True,stdout=subprocess.DEVNULL)
        r=json.loads((destination/'diagnostic.json').read_text())
        if not r['stage1_all20_complete']:raise ValueError('Missing paired20 episodes')
        if any((run/'infrastructure_failures').glob('*.json')):raise ValueError('Unresolved infrastructure failure')
        return {'policies':r['policies'],'paired':r['paired'],'complete_episodes':len(r['episodes'])}
    try:
        if a.wait_pid:
            while Path(f'/proc/{a.wait_pid}').exists():
                info=Path(f'/proc/{a.wait_pid}/stat').read_text().split()
                if info[2]=='Z':break
                cmdline=Path(f'/proc/{a.wait_pid}/cmdline').read_bytes()
                if b'run_decision_rag_stages.py' not in cmdline:raise ValueError('Wait PID identity changed')
                status('waiting_for_budget256_collector',collector_pid=a.wait_pid);time.sleep(10)
        old=root/'results/decision_rag_20261015'
        details=diagnose_report(old,reports/'budget256')
        checkpoint(old,'stage1_budget256_complete',{**details,'next':'common1024_diagnostic'})
        argsfile=repo/'lab/registry/decision_rag_20261015/common_args_1024.json';common=json.loads(argsfile.read_text());new=Path(common[common.index('--output')+1])
        command([sys.executable,'-u','scripts/run_decision_rag_stages.py',*common,'--phase','diagnose','--start','0','--limit','20'],'diagnostic_1024_000_020.log')
        details=diagnose_report(new,reports/'stage1')
        checkpoint(new,'stage1_complete',{**details,'output_budget':1024,'next':'fresh_coverage_gated_teacher_head_dev'})
        command([sys.executable,'-u','scripts/continue_decision_rag.py','--common-args',str(argsfile)],'stage2_coordinator.log')
        result=json.loads((new/'coordinator_status.json').read_text())
        status(result['phase'],run=str(new))
    except BaseException as error:
        status('pipeline_failed',error=type(error).__name__+': '+str(error));raise

if __name__=='__main__':main()
