"""Evaluate fixed constrained candidates and quality starting head serially."""
import json,os,subprocess,time
from pathlib import Path
from run_decision_policy_pilot import ROOT,REPO,save,sha
NAME='decision_matched_refresh_20261024'
VALIDATION='decision_matched_refresh_validation64_20261024'

def read(p):return json.loads(p.read_text())

def main():
    state=ROOT/'results/decision_matched_refresh_coordinator_r1_20261024.json'
    assert not state.exists(),'No implicit restart'
    start=time.monotonic()
    def phase(name,**kw):
        data=dict(phase=name,pid=None if name in ['complete','failed'] else os.getpid(),seconds=time.monotonic()-start,**kw)
        save(state,data)
        p=REPO/'memory/EXPERIMENT_STATUS.json';m=read(p);m['matched_refresh_evaluation']=data;save(p,m)
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
        upstream=ROOT/'results/decision_selected_refresh_coordinator_20261024.json'
        phase('waiting_for_verified_collection',upstream=str(upstream))
        deadline=time.monotonic()+4200
        while True:
            current=read(upstream)
            if current['phase']=='complete':break
            if current['phase']=='failed':raise RuntimeError('Upstream collection failed; no training')
            assert time.monotonic()<deadline,'Collection wait deadline exceeded'
            pid=current.get('pid')
            assert pid,'Missing upstream coordinator handle'
            try:command=Path(f'/proc/{pid}/cmdline').read_bytes()
            except FileNotFoundError:
                # Upstream can finish between the status read and the process check.
                time.sleep(1)
                terminal=read(upstream)
                if terminal['phase']=='complete':break
                raise RuntimeError('Upstream handle exited without successful completion')
            assert b'coordinate_decision_selected_refresh.py' in command,'Upstream process absent or reused'
            time.sleep(10)
        run('scripts/fit_decision_matched_refresh.py',[],600)
        assert read(ROOT/'results'/NAME/'status.json')['phase']=='complete'
        fitted=read(ROOT/'results'/NAME/'manifest.json')
        template=read(REPO/'lab/registry/decision_frozen_adapters_20261024/validation_spec.json')
        template['run_name']=VALIDATION
        template['policies']={'selected_reference':template['policies']['selected_reference'],**fitted['candidates']}
        template['max_seconds']=3000
        template['max_generated']=1000000
        template['max_inputs']=40000000
        for source in ['scripts/evaluate_decision_direct_utility.py','src/lsdrag/decision_direct_utility.py']:
            template['source_sha256'][source]=sha(REPO/source)
        save(REPO/'lab/registry'/NAME/'validation_spec.json',template)
        run('scripts/reporting/archive_decision_run.py',['--run',NAME],300)
        spec=REPO/'lab/registry'/NAME/'validation_spec.json'
        run('scripts/evaluate_decision_direct_utility.py',['--spec',str(spec)],3300)
        run('scripts/reporting/summarize_decision_shared_evaluation.py',['--run',VALIDATION,'--baseline','selected_reference'],180)
        run('scripts/reporting/compare_decision_policy_campaign.py',['--run',VALIDATION],180)
        run('scripts/reporting/archive_decision_run.py',['--run',VALIDATION],300)
        phase('complete',report=f'lab/reports/{VALIDATION}/report.md',goal_achieved=False)
    except BaseException as e:phase('failed',error=repr(e));raise
if __name__=='__main__':main()
