"""Await this collection only; audit and fit once, never advance to actor training."""
import json,os,subprocess,time
from pathlib import Path
from run_decision_policy_pilot import ROOT,REPO,save,sha
NAME='decision_future_critic_probe72_20261024'
def main():
    state=ROOT/'results'/(NAME+'_coordinator.json');assert not state.exists()
    collection=ROOT/'results/decision_h3_diversity48_coordinator_20261024.json'
    def status(phase,**kw):save(state,dict(phase=phase,pid=None if phase in ['complete','failed'] else os.getpid(),**kw))
    def run(script,args=()):
        log=ROOT/'logs'/f'{NAME}_{Path(script).stem}.log'
        with log.open('x') as f:
            p=subprocess.Popen([str(ROOT/'runtime/bin/python'),script,*args],cwd=REPO,env=dict(os.environ,PYTHONPATH='src:scripts'),stdout=f,stderr=subprocess.STDOUT)
            status('running',child_pid=p.pid,script=script,log=str(log))
            try:code=p.wait(timeout=600)
            except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30);raise
            assert code==0,f'{script} failed: {log}'
    try:
        deadline=time.monotonic()+3900
        while True:
            d=json.loads(collection.read_text())
            if d['phase']=='complete':break
            assert d['phase']!='failed',d
            pid=d['pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
            assert b'coordinate_h3_diversity48.py' in cmd,'Expected coordinator absent; no restart'
            assert time.monotonic()<deadline,'Timed out awaiting collection'
            status('waiting',collection_pid=pid);time.sleep(15)
        run('scripts/reporting/verify_h3_diversity48.py')
        p=REPO/'lab/registry'/NAME/'plan.json';m=json.loads(p.read_text())
        assert m['pending_completed_collection_hash']
        m['input_sha256'][m['inputs'][1]]=sha(Path(m['inputs'][1]));m['pending_completed_collection_hash']=False;save(p,m)
        run('scripts/probe_decision_future_critic72.py')
        run('scripts/reporting/archive_decision_run.py',['--run',NAME])
        status('complete',next='Independently verify critic comparison; no actor auto-launch')
    except BaseException as e:status('failed',error=repr(e));raise
if __name__=='__main__':main()
