"""Bounded teacher completion -> CPU fit -> same exposed calibration16."""
import json
import os
from pathlib import Path
import subprocess
import time
from run_decision_policy_pilot import ROOT, REPO, save, sha

def main():
    state = ROOT/'results/decision_shared_initialization_coordinator_20261023.json'
    teacher = ROOT/'results/decision_shared_teacher192_20261023'
    start = time.monotonic()
    def phase(name, **kw):
        save(state, dict(phase=name, pid=None if name in ['complete','failed'] else os.getpid(), **kw))
    try:
        phase('waiting_teacher')
        while True:
            status = json.loads((teacher/'status.json').read_text())
            if status['phase']=='complete':
                break
            if status['phase']=='failed':
                raise RuntimeError('Teacher collection failed; no restart')
            proc = Path(f'/proc/{status["pid"]}/cmdline')
            if not proc.exists() or b'collect_decision_shared_teacher.py' not in proc.read_bytes():
                raise RuntimeError('Teacher process missing; inspect artifacts')
            if time.monotonic()-start > 3900:
                raise TimeoutError('Teacher wait budget')
            time.sleep(15)
        def run(script, args, timeout):
            log = ROOT/'logs'/f'{Path(script).stem}_20261023.log'
            with log.open('x') as output:
                child = subprocess.Popen([str(ROOT/'runtime/bin/python'),script,*args], cwd=REPO, env=dict(os.environ,PYTHONPATH='src',OPENBLAS_NUM_THREADS='2'),stdout=output,stderr=subprocess.STDOUT)
                phase('running', script=script, child_pid=child.pid, log=str(log))
                try:
                    code = child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    child.terminate();child.wait(timeout=30)
                    raise
                if code:
                    raise RuntimeError(f'{script} failed: {log}')
        run('scripts/fit_decision_shared_initialization.py', [], 300)
        spec = json.loads((REPO/'lab/registry/decision_shared_calibration16_20261023/spec.json').read_text())
        head = json.loads((ROOT/'results/decision_shared_initialization_20261023/manifest.json').read_text())
        spec.update(run_name='decision_shared_initialization_validation16_20261023', policies={'shared_bc':{k:head[k] for k in ['kind','checkpoint','sha256']}},max_seconds=900)
        spec['source_sha256']={p:sha(REPO/p) for p in spec['source_sha256']}
        path = REPO/'lab/registry/decision_shared_teacher192_20261023/initialization_validation_spec.json'
        assert not path.exists()
        save(path,spec)
        run('scripts/evaluate_decision_shared_corpus.py',['--spec',str(path)],1200)
        phase('complete', next='Audit teacher, initialization and calibration. Freeze shared GRPO/branch learning; final test still untouched.')
    except BaseException as exc:
        phase('failed', error=repr(exc))
        raise

if __name__=='__main__':
    main()
