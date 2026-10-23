"""Wait for main comparison, then run one cached structural candidate pair."""
import json
import os
from pathlib import Path
import subprocess
import time
from run_decision_policy_pilot import ROOT, REPO, save, sha

def read(path):return json.loads(path.read_text())

def main():
    state=ROOT/'results/decision_shared_structure_coordinator_20261023.json'
    upstream=ROOT/'results/decision_shared_learning_coordinator_20261023.json'
    started=time.monotonic()
    def phase(name,**kw):save(state,dict(phase=name,pid=None if name in ['complete','failed'] else os.getpid(),**kw))
    def run(script,args=(),timeout=300):
        log=ROOT/'logs'/f'shared_structure_{Path(script).stem}_{int(time.time())}.log'
        with log.open('x') as output:
            child=subprocess.Popen([str(ROOT/'runtime/bin/python'),script,*args],cwd=REPO,env=dict(os.environ,PYTHONPATH='src',OPENBLAS_NUM_THREADS='2'),stdout=output,stderr=subprocess.STDOUT)
            phase('running',script=script,child_pid=child.pid,log=str(log))
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=30);raise
            if code:raise RuntimeError(f'{script} failed: {log}')
    try:
        phase('waiting_main_comparison')
        while True:
            current=read(upstream)
            if current['phase']=='complete':break
            if current['phase']=='failed':raise RuntimeError('Main comparison failed; inspect before continuing')
            proc=Path(f'/proc/{current["pid"]}/cmdline')
            if not proc.exists() or b'coordinate_decision_shared_learning.py' not in proc.read_bytes():raise RuntimeError('Main coordinator missing; no automatic restart')
            if time.monotonic()-started>14400:raise TimeoutError('Four-hour upstream wait cap')
            time.sleep(15)
        plan=read(REPO/'lab/registry/decision_shared_structure_20261023/plan.json')
        run('scripts/fit_decision_shared_structure.py',timeout=plan['max_cpu_seconds'])
        bank=read(ROOT/'results/decision_shared_structure_20261023/manifest.json')
        spec=read(REPO/'lab/registry/decision_shared_outcomes192_20261023/validation_spec.json')
        spec.update(run_name='decision_shared_structure_validation64_20261023',policies=bank['candidates'],max_seconds=plan['max_evaluation_seconds'],selection='Frozen multi-position MLP warmstart/outcome pair; compare to main validation through explicit per-run provenance. No additional collection or final test access.')
        spec['source_sha256']={p:sha(REPO/p) for p in spec['source_sha256']}
        path=REPO/'lab/registry/decision_shared_structure_20261023/validation_spec.json'
        assert not path.exists();save(path,spec)
        run('scripts/evaluate_decision_shared_corpus.py',['--spec',str(path)],timeout=plan['max_evaluation_seconds']+180)
        run('scripts/reporting/summarize_decision_shared_evaluation.py',['--run',spec['run_name'],'--baseline','multi_mlp_bc'],timeout=180)
        phase('complete',next='Audit/cache backups, consolidate validation against shared_grpo using explicit run provenance, freeze final candidates before test160. Within-pair report is not the main GRPO target test.')
    except BaseException as exc:
        phase('failed',error=repr(exc));raise

if __name__=='__main__':main()
