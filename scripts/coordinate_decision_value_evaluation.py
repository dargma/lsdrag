"""Wait for the live zero-gate coordinator; evaluate fixed candidates serially."""
import json,os,subprocess,time
from pathlib import Path
from run_decision_policy_pilot import ROOT,REPO,save
NAME='decision_opportunity_value_20261024'
VALIDATION='decision_opportunity_value_validation64_20261024'

def read(p):return json.loads(p.read_text())

def main():
    state=ROOT/'results/decision_opportunity_value_coordinator_20261024.json'
    assert not state.exists(),'No implicit restart'
    start=time.monotonic()
    def phase(name,**kw):
        data=dict(phase=name,pid=None if name in ['complete','failed'] else os.getpid(),seconds=time.monotonic()-start,**kw)
        save(state,data)
        p=REPO/'memory/EXPERIMENT_STATUS.json';m=read(p);m['opportunity_value_evaluation']=data;save(p,m)
    def run(script,args,timeout):
        log=ROOT/'logs'/f'value_heads_{Path(script).stem}_{int(time.time())}.log'
        with log.open('x') as f:
            child=subprocess.Popen([str(ROOT/'runtime/bin/python'),script,*args],cwd=REPO,env=dict(os.environ,PYTHONPATH='src',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
            phase('running',script=script,child_pid=child.pid,log=str(log))
            try:rc=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
            if rc:raise RuntimeError(f'{script} exited {rc}; see {log}')
    try:
        assert read(ROOT/'results'/NAME/'status.json')['phase']=='complete'
        run('scripts/reporting/archive_decision_run.py',['--run',NAME],300)
        predecessor=ROOT/'results/decision_selected_head_grpo_coordinator_20261024_zero_gate.json'
        while True:
            previous=read(predecessor)
            if previous['phase']=='complete':break
            if previous['phase']=='failed':raise RuntimeError('Predecessor failed; inspect before follow-up evaluation')
            pid=previous['pid'];os.kill(pid,0)
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
            assert b'coordinate_decision_selected_head_grpo_zero_gate.py' in cmd,'Predecessor PID reused'
            assert time.monotonic()-start<5400,'Wait budget exhausted; do not restart predecessor'
            phase('waiting',predecessor_pid=pid,predecessor=str(predecessor))
            time.sleep(30)
        spec=REPO/'lab/registry'/NAME/'validation_spec.json'
        run('scripts/evaluate_decision_value_heads.py',['--spec',str(spec)],2100)
        run('scripts/reporting/summarize_decision_shared_evaluation.py',['--run',VALIDATION,'--baseline','mlp_control'],180)
        run('scripts/reporting/archive_decision_run.py',['--run',VALIDATION],300)
        phase('complete',report=f'lab/reports/{VALIDATION}/report.md',goal_achieved=False)
    except BaseException as e:phase('failed',error=repr(e));raise
if __name__=='__main__':main()
