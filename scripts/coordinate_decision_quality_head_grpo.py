"""Bounded continuation of the selected head, followed by paired validation only."""
import json
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone
from run_decision_policy_pilot import ROOT, REPO, save, sha

NAME = 'decision_quality_head_grpo192_20261024'
VALIDATION = 'decision_quality_head_grpo_validation64_20261024'
REG = REPO / 'lab/registry' / NAME


def read(p):
    return json.loads(p.read_text())


def main():
    state = ROOT / 'results/decision_quality_head_grpo_coordinator_20261024.json'
    assert not state.exists(), 'No implicit restart'
    started = time.monotonic()

    def phase(name, **kw):
        data = dict(phase=name, pid=None if name in ('complete', 'failed') else os.getpid(),
                    seconds=time.monotonic()-started, **kw)
        save(state, data)
        now = datetime.now(timezone.utc).isoformat()
        note = (f'## {now} — selected-head GRPO continuation: {name}\n\n'
                f'Coordinator state: `{state}`. Details: `{json.dumps(data)}`. '
                'Same already-trained selected head continues learning; LLM/normalization fixed. '
                'Only train192 and validation64 used; final80/reserved80 protected.\n\n')
        for filename in ('CURRENT.md', 'HEAD_EXPERIMENT.md'):
            path = REPO / 'memory' / filename
            path.write_text(note + path.read_text())
        path = REPO / 'memory/SESSION_LOG.md'
        path.write_text(path.read_text() + '\n' + note)
        path = REPO / 'memory/EXPERIMENT_STATUS.json'
        memory = read(path)
        memory['quality_head_grpo_continuation'] = data | {'updated_utc': now, 'run': NAME}
        memory.update(phase='quality_head_grpo_' + name, pid=kw.get('child_pid'),
                      updated_utc=now, next='Finish bounded train/audit/paired validation/archive. No test launch.')
        memory['training_completed'] = (ROOT/'results'/NAME/'summary.json').exists()
        evaluation_status = ROOT/'results'/VALIDATION/'status.json'
        memory['evaluation_completed'] = evaluation_status.exists() and read(evaluation_status)['phase']=='complete'
        memory['run'] = str(ROOT/'results'/NAME)
        save(path, memory)

    def run(script, args=(), timeout=300):
        log = ROOT / 'logs' / f'selected_grpo_{Path(script).stem}_{int(time.time())}.log'
        with log.open('x') as stream:
            child = subprocess.Popen([str(ROOT/'runtime/bin/python'), script, *args], cwd=REPO,
                env=dict(os.environ, PYTHONPATH='src', OPENBLAS_NUM_THREADS='2'),
                stdout=stream, stderr=subprocess.STDOUT)
            phase('running', script=script, child_pid=child.pid, log=str(log))
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                raise
            if code:
                raise RuntimeError(f'{script} exited {code}; inspect {log}')

    try:
        phase('preparing')
        assert not (ROOT/'results'/NAME).exists()
        predecessor=ROOT/'results/decision_current_policy_refresh_coordinator_20261024.json'
        while True:
            previous=read(predecessor)
            if previous['phase']=='complete':break
            if previous['phase']=='failed':raise RuntimeError('Refresh failed; inspect before GRPO')
            os.kill(previous['pid'],0)
            assert b'coordinate_decision_current_refresh.py' in Path(f"/proc/{previous['pid']}/cmdline").read_bytes()
            assert time.monotonic()-started<4200
            save(state,dict(phase='waiting',pid=os.getpid(),predecessor_pid=previous['pid'],seconds=time.monotonic()-started))
            time.sleep(30)
        run('scripts/run_decision_quality_head_grpo.py', timeout=3900)
        run('scripts/reporting/audit_decision_quality_head_grpo.py', timeout=180)
        run('scripts/reporting/archive_decision_run.py', ['--run', NAME], timeout=300)
        manifest = read(REG/'manifest.json')
        spec = read(REG/'validation_template.json')
        cp = ROOT/'results'/NAME/f'grpo_iteration{manifest["iterations"]-1}.pt'
        spec['policies'] = {'quality_start': manifest['reference_spec'],
            'quality_grpo': dict(kind='decision_value', body='mlp',
                                         checkpoint=str(cp), sha256=sha(cp))}
        # All runtime source hashes were frozen before training; evaluator checks them.
        path = REG/'validation_spec.json'
        assert not path.exists()
        save(path, spec)
        run('scripts/evaluate_decision_value_heads.py', ['--spec', str(path)], timeout=1500)
        run('scripts/reporting/summarize_decision_shared_evaluation.py',
            ['--run', VALIDATION, '--baseline', 'quality_start'], timeout=180)
        run('scripts/reporting/compare_decision_policy_campaign.py', ['--run', VALIDATION], timeout=180)
        run('scripts/reporting/archive_decision_run.py', ['--run', VALIDATION], timeout=300)
        phase('complete', report=f'lab/reports/{VALIDATION}/report.md',
              next='Review paired validation; no original-final80/reserved80 evaluation or new sweep.')
    except BaseException as exc:
        phase('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
