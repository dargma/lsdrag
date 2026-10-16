"""Local 30-minute GPU gate that resumes a pinned existing Codex session once.

At each due time, require five minutes of idle samples before starting.
No GPU allocation until the due time. Unknown telemetry is busy. Survives chat
completion, but not host shutdown. Never terminates another process.
"""
import argparse
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def sample(gpu):
    def query(fields, kind):
        return subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-'+kind+'='+fields,
            '--format=csv,noheader,nounits'], text=True, timeout=15)
    apps = list(csv.reader(io.StringIO(query('pid,process_name,used_gpu_memory', 'compute-apps'))))
    apps = [r for r in apps if r]
    rows = list(csv.reader(io.StringIO(query('uuid,utilization.gpu,memory.used,memory.free', 'gpu'))))
    if len(rows) != 1 or len(rows[0]) != 4 or rows[0][0].strip() != gpu:
        raise ValueError('Unexpected GPU telemetry')
    util, used, free = [float(x.strip()) for x in rows[0][1:]]
    if not all(math.isfinite(x) and x >= 0 for x in [util, used, free]):
        raise ValueError('Invalid GPU telemetry')
    return {'compute_apps': apps, 'utilization': util, 'used_mib': used, 'free_mib': free,
            'idle': not apps and util <= 5 and used <= 1024 and free >= 32768}


def confirm_idle(gpu, seconds, poll, cancelled, observe):
    """Require idle observations over the full window; gaps reset evidence."""
    since = previous = time.monotonic()
    while not cancelled():
        target = time.monotonic()+poll
        while time.monotonic() < target:
            if cancelled(): return False
            time.sleep(min(1, target-time.monotonic()))
        info = sample(gpu)
        now = time.monotonic()
        if now-previous > 65:
            since = now
        previous = now
        observe(info, now-since)
        if not info['idle']: return False
        if now-since >= seconds: return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    root = Path(spec['state_dir']); root.mkdir(parents=True, exist_ok=True)
    lock = (root/'watch.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root/'launch_intent.json').exists():
        raise RuntimeError('Already claimed; inspect before any relaunch')
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def save(phase, **kw):
        data = {'phase': phase, 'pid': os.getpid(), 'updated_epoch': time.time(), **kw}
        tmp = root/'status.tmp'; tmp.write_text(json.dumps(data, indent=2)+'\n');tmp.replace(root/'status.json')
    due = spec['first_check_epoch']
    save('waiting_first_check', next_check_epoch=due)
    while not stopped and not (root/'STOP').exists():
        if time.time() < due:
            time.sleep(min(1, due-time.time())); continue
        try:
            info = sample(spec['gpu_uuid'])
            with (root/'checks.jsonl').open('a') as f:
                f.write(json.dumps({'time': time.time(), **info})+'\n')
            # Recheck immediately before handing control to the experiment session.
            if info['idle']:
                def observe_idle(info, elapsed):
                    with (root/'checks.jsonl').open('a') as f:
                        f.write(json.dumps({'time': time.time(), 'idle_seconds': elapsed, **info})+'\n')
                    save('confirming_five_minutes_idle', idle_seconds=elapsed, gpu=info)
                confirmed = confirm_idle(spec['gpu_uuid'], spec['idle_seconds'],
                    spec['idle_poll_seconds'], lambda: stopped or (root/'STOP').exists(), observe_idle)
                if not confirmed:
                    if stopped or (root/'STOP').exists(): break
                    due += spec['interval_seconds']
                    while due <= time.time(): due += spec['interval_seconds']
                    save('idle_window_interrupted', next_check_epoch=due)
                    continue
                info = sample(spec['gpu_uuid'])
                if info['idle']:
                    intent = {'time': time.time(), 'command': spec['command'], 'gpu': info}
                    with (root/'launch_intent.json').open('x') as f: json.dump(intent, f, indent=2)
                    with (root/'executor.log').open('a') as log:
                        proc = subprocess.Popen(spec['command'], cwd=spec['cwd'], stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL)
                        save('executor_running', child_pid=proc.pid, thread_id=spec['thread_id'])
                        # STOP after launch stops the monitor only, never the executor or other jobs.
                        while proc.poll() is None:
                            if stopped or (root/'STOP').exists():
                                save('monitor_stopped_executor_running', child_pid=proc.pid);return
                            time.sleep(1)
                    save('executor_completed' if proc.returncode == 0 else 'executor_failed',
                         child_pid=proc.pid, exit_code=proc.returncode)
                    return
            due += spec['interval_seconds']
            while due <= time.time(): due += spec['interval_seconds']
            save('waiting_gpu_idle', next_check_epoch=due, gpu=info)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            due = max(due+spec['interval_seconds'], time.time()+spec['interval_seconds'])
            save('check_error_no_launch', next_check_epoch=due, error=str(exc))
    save('stopped')


if __name__ == '__main__':
    main()
