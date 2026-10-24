"""Bounded current-policy branch collection and raw consistency audit/archive."""
import json,os,subprocess,time
from pathlib import Path
import torch
from run_decision_policy_pilot import ROOT,REPO,save,sha
NAME='decision_stop_value_h2_collect48_20261024'
def read(p):return json.loads(p.read_text())
def main():
    state=ROOT/'results/decision_stop_value_h2_collect_coordinator_20261024.json';assert not state.exists()
    start=time.monotonic()
    def phase(name,**kw):
        data=dict(phase=name,pid=None if name in ['complete','failed'] else os.getpid(),seconds=time.monotonic()-start,**kw);save(state,data)
        p=REPO/'memory/EXPERIMENT_STATUS.json';m=read(p);m['stop_value_h2_collection']=data;save(p,m)
    def run(script,args,timeout):
        log=ROOT/'logs'/f'refresh48_{Path(script).stem}_{int(time.time())}.log'
        with log.open('x') as f:
            child=subprocess.Popen([str(ROOT/'runtime/bin/python'),script,*args],cwd=REPO,env=dict(os.environ,PYTHONPATH='src',OPENBLAS_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
            phase('running',child_pid=child.pid,script=script,log=str(log))
            try:rc=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
            if rc:raise RuntimeError(f'{script} exited {rc}: {log}')
    try:
        run('scripts/run_decision_stop_value_h2_collect.py',[],2100)
        p=ROOT/'results'/NAME;m=read(p/'manifest.json');s=read(p/'status.json');assert s['phase']=='complete'
        records=torch.load(p/'states.pt',weights_only=True,map_location='cpu')
        eps=[read(x) for x in (p/'episodes').glob('*.json')];by={(e['qid'],e['tag']):e for e in eps}
        assert len(by)==len(eps)==s['episodes']
        assert {r['qid'] for r in records}==set(m['train']) and set(m['train'])<=set(m['train_pool'])
        assert not set(m['train'])&set(m['validation']+m['test'])
        for field,key in [('generated_tokens','generated'),('prompt_tokens','inputs')]:assert sum(e[field] for e in eps)==s[key]
        for r in records:
            assert r['pack'].dtype==torch.float32
            for a in range(4):
                samples=[v for v in r['samples'] if v['action']==a]
                assert len(samples)==(1 if a in [3,r['continue_action']] else 0)
                for v in samples:
                    e=by[r['qid'],f"t{r['turn']}_a{a}_r{v['rep']}"]
                    assert e['f1']==v['f1'] and e['generated_tokens']==v['generated']
                    assert e['decisions'][0]['action']==a and e['decisions'][0]['branch_intervention']
                    assert e['reader_calls'][0]['messages_sha256']==r['prefix_sha256']
        report=dict(audit_passed=True,questions=48,states=len(records),episodes=len(eps),counts=s,states_sha256=sha(p/'states.pt'),errors=sum(bool(e['errors']) for e in eps),note='Collection consistency only. Fresh current-policy suffix outcomes for train subset; no validation/test quality claim.')
        dest=REPO/'lab/reports'/NAME;dest.mkdir(parents=True,exist_ok=True);save(dest/'audit.json',report)
        run('scripts/reporting/archive_decision_run.py',['--run',NAME],300)
        phase('complete',report=str(dest/'audit.json'),goal_achieved=False)
    except BaseException as e:phase('failed',error=repr(e));raise
if __name__=='__main__':main()
