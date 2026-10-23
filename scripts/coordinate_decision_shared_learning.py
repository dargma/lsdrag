"""Frozen sequential learning and validation; no final-test auto-launch."""
import json
import os
from pathlib import Path
import subprocess
import time
from run_decision_policy_pilot import ROOT, REPO, save, sha

def read(path):
    return json.loads(path.read_text())

def main():
    state = ROOT/'results/decision_shared_learning_coordinator_20261023.json'
    start = time.monotonic()
    def phase(name, **kw):
        save(state, dict(phase=name, pid=None if name in ['complete','failed'] else os.getpid(), seconds=time.monotonic()-start, **kw))
    def run(script, args=(), timeout=300):
        log = ROOT/'logs'/f'shared_learning_{Path(script).stem}_{int(time.time())}.log'
        with log.open('x') as output:
            child = subprocess.Popen([str(ROOT/'runtime/bin/python'),script,*args],cwd=REPO,env=dict(os.environ,PYTHONPATH='src',OPENBLAS_NUM_THREADS='2'),stdout=output,stderr=subprocess.STDOUT)
            phase('running',script=script,child_pid=child.pid,log=str(log))
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=30)
                raise
            if code:
                raise RuntimeError(f'{script} exited {code}; inspect {log}')
    try:
        phase('checking_frozen_inputs')
        for name in ['decision_shared_grpo192_20261023','decision_shared_outcomes192_20261023']:
            assert (REPO/'lab/registry'/name/'manifest.json').is_file()
            assert not (ROOT/'results'/name).exists()
        run('scripts/run_decision_shared_grpo.py',timeout=3900)
        run('scripts/run_decision_shared_outcomes.py',timeout=7500)
        run('scripts/reporting/audit_decision_state_outcomes.py',['--run','decision_shared_outcomes192_20261023'],timeout=180)
        run('scripts/fit_decision_shared_outcomes.py',timeout=300)
        heads = ROOT/'results/decision_shared_heads_20261023'
        bank = read(heads/'manifest.json')
        checkpoint = ROOT/'results/decision_shared_grpo192_20261023/grpo_iteration1.pt'
        bank['candidates']['shared_grpo'] = dict(kind='original',checkpoint=str(checkpoint),sha256=sha(checkpoint))
        save(heads/'manifest.json',bank)
        spec = read(REPO/'lab/registry/decision_shared_calibration16_20261023/spec.json')
        manifest = read(REPO/'lab/registry/decision_shared_outcomes192_20261023/manifest.json')
        spec.update(run_name='decision_shared_validation64_20261023',qids=manifest['validation'],selection='All previously frozen validation64, with original32 and new32 reported separately. No final-test access.',policies={'semantic_once':{'kind':'rule'},'native_fast':{'kind':'native'},**bank['candidates']},max_seconds=1800,max_generated=1800000,max_inputs=60000000)
        spec['source_sha256']={p:sha(REPO/p) for p in spec['source_sha256']}
        path=REPO/'lab/registry/decision_shared_outcomes192_20261023/validation_spec.json'
        assert not path.exists()
        save(path,spec)
        run('scripts/evaluate_decision_shared_corpus.py',['--spec',str(path)],timeout=2100)
        run('scripts/reporting/summarize_decision_shared_evaluation.py',['--run',spec['run_name']],timeout=180)
        phase('complete',next='Audit GRPO/checkpoint changes, archive training/validation/head bank, report original vs new validation32, select candidates before final160. No final test launched.')
    except BaseException as exc:
        phase('failed',error=repr(exc))
        raise

if __name__=='__main__':
    main()
