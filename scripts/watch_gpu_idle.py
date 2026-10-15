"""One-shot local GPU idle gate. Survives chat end, not host shutdown.

Launches only the argv pinned in a reviewed JSON spec; never uses a shell or
terminates other GPU jobs. Restarting the monitor resets its idle interval.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


class IdleWindow:
    def __init__(self, seconds=1200, max_gap=65):
        self.seconds, self.max_gap = seconds, max_gap
        self.since = self.previous = None

    def observe(self, now, eligible):
        if self.previous is not None and (now < self.previous or now-self.previous > self.max_gap):
            self.since = None
        self.previous = now
        if not eligible:
            self.since = None
        elif self.since is None:
            self.since = now
        elapsed = 0 if self.since is None else now-self.since
        return elapsed >= self.seconds, elapsed


def sample_gpu(gpu_uuid):
    output = subprocess.check_output(['nvidia-smi', '-i', gpu_uuid,
        '--query-gpu=uuid,utilization.gpu,utilization.memory,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = [line.split(',') for line in output.strip().splitlines()]
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError('Unexpected GPU telemetry')
    uuid, util, mem_util, free = (value.strip() for value in rows[0])
    values = [float(value) for value in (util, mem_util, free)]
    if uuid != gpu_uuid or not all(math.isfinite(x) for x in values):
        raise ValueError('Invalid GPU identity/telemetry')
    return dict(uuid=uuid, utilization=values[0], memory_utilization=values[1], free_mib=values[2])


def verify_spec(spec):
    cwd = Path(spec['cwd']).resolve()
    if not spec['command'] or not all(isinstance(v, str) for v in spec['command']):
        raise ValueError('Expected explicit argv')
    if not spec.get('source_sha256'):
        raise ValueError('Missing source pins')
    for name, digest in spec['source_sha256'].items():
        path = cwd/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Source changed: '+name)
    for package, version in spec.get('packages', {}).items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Runtime package changed: '+package)
    return cwd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    cwd = verify_spec(spec)
    root = args.state_dir
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root/'launch_intent.json').exists():
        raise ValueError('One-shot launch already claimed; inspect existing run, do not relaunch')
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def status(phase, **extra):
        value = dict(phase=phase, pid=os.getpid(), updated_epoch=time.time(), **extra)
        temp = root/'status.tmp'
        temp.write_text(json.dumps(value, indent=2)+'\n')
        temp.replace(root/'status.json')
        return value
    window = IdleWindow()
    status('watching', idle_seconds=0, spec=str(args.spec.resolve()))
    while not stopped and not (root/'STOP').exists():
        try:
            gpu = sample_gpu(spec['gpu_uuid'])
            eligible = (gpu['utilization'] <= 5 and gpu['memory_utilization'] <= 5
                        and gpu['free_mib'] >= spec['min_free_gib']*1024)
            ready, elapsed = window.observe(time.monotonic(), eligible)
            status('watching', gpu=gpu, idle_seconds=elapsed, eligible=eligible)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            window.observe(time.monotonic(), False)
            status('telemetry_unavailable', error=str(error), idle_seconds=0)
            ready = False
        if ready:
            try:
                verify_spec(spec)
                # Recheck immediately before claiming a launch. No GPU reservation
                # is implied: the runner checks free VRAM again before loading.
                gpu = sample_gpu(spec['gpu_uuid'])
                if gpu['utilization'] > 5 or gpu['memory_utilization'] > 5 or gpu['free_mib'] < spec['min_free_gib']*1024:
                    window.observe(time.monotonic(), False)
                    continue
                if stopped or (root/'STOP').exists():
                    break
                intent = dict(command=spec['command'], cwd=str(cwd), time=time.time(), gpu=gpu)
                with (root/'launch_intent.json').open('x') as stream:
                    json.dump(intent, stream, indent=2)
                with (root/'experiment.log').open('a') as log:
                    proc = subprocess.Popen(spec['command'], cwd=cwd, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                        env={**os.environ, 'PYTHONPATH': str(cwd/'src'), 'CUDA_VISIBLE_DEVICES': spec['gpu_uuid']})
                    status('running', child_pid=proc.pid, command=spec['command'])
                    while proc.poll() is None:
                        if stopped or (root/'STOP').exists():
                            os.killpg(proc.pid, signal.SIGTERM)
                            try:
                                proc.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                            break
                        time.sleep(1)
                status('stopped' if stopped or (root/'STOP').exists() else ('completed' if proc.returncode == 0 else 'experiment_failed'),
                       child_pid=proc.pid, exit_code=proc.returncode)
            except Exception as error:
                status('launch_blocked', error=type(error).__name__+': '+str(error))
            return
        for _ in range(30):
            if stopped or (root/'STOP').exists():
                break
            time.sleep(1)
    status('stopped')


if __name__ == '__main__':
    main()
