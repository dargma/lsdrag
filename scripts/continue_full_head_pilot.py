"""Continue the frozen head-only stages after an existing first20 collector exits."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import torch

from lsdrag.io import dump_json, read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--common-args', type=Path, required=True)
    p.add_argument('--wait-pid', type=int)
    p.add_argument('--backup', type=Path, required=True)
    args = p.parse_args()
    common = json.loads(args.common_args.read_text())
    value = lambda key: Path(common[common.index('--'+key)+1])
    run, data = value('output'), value('train-questions').parent
    repo = Path(__file__).resolve().parents[1]
    logdir = run.parent.parent/'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    questions = read_jsonl(value('train-questions'))
    actions = json.loads((run/'manifest.json').read_text())['feature_contract']['actions']

    def publish(tag):
        # User explicitly requested cumulative Git memory at every phase.
        memory = repo/'memory'
        memory.mkdir(exist_ok=True)
        record = json.loads((run/'coordinator_status.json').read_text())
        dump_json(memory/'EXPERIMENT_STATUS.json', {**record, 'run': str(run), 'world_model': 'deferred'})
        with (memory/'SESSION_LOG.md').open('a') as stream:
            stream.write('\n## '+time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())+' — automated head checkpoint\n\n'+json.dumps(record, ensure_ascii=False)+'\n')
        report = repo/'lab/reports/full_head_20261015_v2'
        selected = ['memory/EXPERIMENT_STATUS.json', 'memory/SESSION_LOG.md']
        selected += [str(path.relative_to(repo)) for path in report.glob('*') if path.suffix in {'.json', '.md'}]
        for command in (['git', 'add']+selected,
                        ['git', '-c', 'user.name=Codex', '-c', 'user.email=codex@users.noreply.github.com',
                         'commit', '-m', 'Record automated full-head phase '+tag],
                        ['git', 'push', 'origin', 'jev_v1.1']):
            outcome = subprocess.run(command, cwd=repo, capture_output=True, text=True)
            if outcome.returncode:
                with (run/'memory_publish_errors.log').open('a') as stream:
                    stream.write(time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())+' '+str(command[:2])+' '+outcome.stderr+'\n')
                print('Memory saved locally; Git publication encountered an error. See memory_publish_errors.log', flush=True)
                break

    def status(phase, **kwargs):
        record = {'phase': phase, 'updated_utc_epoch': time.time(), **kwargs}
        dump_json(run/'coordinator_status.json', record)
        print(json.dumps(record), flush=True)
        if phase in {'completed', 'coverage_no_go', 'execution_failed'}:
            publish(phase)

    def report_and_backup(tag):
        subprocess.run([sys.executable, 'scripts/report_full_head_pilot.py', '--run', str(run),
            '--data', str(data), '--report', str(repo/'lab/reports/full_head_20261015_v2')], check=True, cwd=repo)
        staging = run.parent.parent/'backups'
        staging.mkdir(exist_ok=True)
        archive = staging/(tag+'.tar.gz')
        with tarfile.open(archive, 'w:gz') as tf:
            tf.add(run, arcname=run.name)
            tf.add(data, arcname='data')
        args.backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, args.backup/archive.name)
        status('checkpoint_archived', tag=tag, bytes=archive.stat().st_size)
        publish(tag)

    def completed():
        ids = {q['qid'] for q in questions if (run/'teacher'/f'{q["qid"]}.pt').exists()
               and (run/'teacher'/f'{q["qid"]}.json').exists()}
        if ids != {q['qid'] for q in questions[:len(ids)]}:
            raise ValueError('Completed teacher records are not a contiguous frozen prefix')
        return len(ids)

    def coverage():
        seen = {a: set() for a in actions}
        for q in questions[:completed()]:
            artifact = torch.load(run/'teacher'/f'{q["qid"]}.pt', map_location='cpu', weights_only=True)
            for record in artifact['records']:
                seen[record['action']].add(record['qid'])
        return {a: len(ids) for a, ids in seen.items()}, len(set().union(*seen.values()))

    def execute(phase, start=0, limit=20):
        if phase != 'train':
            while True:
                free = float(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True).splitlines()[0])/1024
                if free >= 32: break
                status('waiting_for_vram', required_gib=32, available_gib=free, next_phase=phase)
                time.sleep(15)
        status('starting_'+phase, start=start, limit=limit)
        log = logdir/f'{run.name}_{phase}_{start:03d}_{start+limit:03d}.log'
        with log.open('w') as stream:
            result = subprocess.run([sys.executable, '-u', 'scripts/run_full_head_pilot.py',
                '--phase', phase, '--start', str(start), '--limit', str(limit)]+common,
                stdout=stream, stderr=subprocess.STDOUT, cwd=repo)
        if result.returncode:
            status('execution_failed', stage=phase, returncode=result.returncode, log=str(log))
            raise SystemExit(result.returncode)

    if args.wait_pid:
        status('waiting_for_first20', pid=args.wait_pid)
        while True:
            try:
                command = Path(f'/proc/{args.wait_pid}/cmdline').read_bytes()
            except FileNotFoundError:
                break
            if b'run_full_head_pilot.py' not in command or str(run).encode() not in command:
                break
            time.sleep(5)
    if any((run/'infrastructure_failures').glob('*.json')):
        raise ValueError('Preserved infrastructure failure requires inspection; no silent retry')
    if completed() != 20:
        raise ValueError('Expected the fixed first20 diagnostic to finish before continuation')
    report_and_backup('v2_teacher_020')
    for target in (60, 80, 100):
        start = completed()
        execute('teacher', start, target-start)
        if completed() != target:
            raise ValueError('Teacher phase did not complete its frozen prefix')
        counts, covered_questions = coverage()
        status('teacher_prefix_completed', questions=target, distinct_questions_per_action=counts, natural_teacher_questions=covered_questions)
        report_and_backup(f'v2_teacher_{target:03d}')
        if (all(counts[a] >= 5 for a in actions) and covered_questions >= 30) or target == 100:
            execute('train')
            trained = json.loads((run/'training.json').read_text())
            if not trained['gate']['passed']:
                status('coverage_no_go', stage2='not_run', gate=trained['gate'])
                report_and_backup('v2_coverage_no_go')
                status('coverage_no_go', stage2='not_run', gate=trained['gate'])
                return
            execute('evaluate')
            report_and_backup('v2_completed')
            status('completed', stage2='trained_and_development_evaluated', seeds=trained['seeds'])
            return


if __name__ == '__main__':
    main()
