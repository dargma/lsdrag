"""CPU-only monitor: require 1200 continuous idle seconds before one pipeline."""
import fcntl,json,os,subprocess,time
from pathlib import Path
REPO=Path('/content/local_fast/lsdrag-code');ROOT=Path('/content/local_fast/lsdrag-text-v1')
REG=REPO/'lab/registry/qwen35_9b_h3_20261024';STATE=ROOT/'results/qwen35_9b_idle_gate_20261024';STATE.mkdir(parents=True,exist_ok=True)
lock=(STATE/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
start=None
plan=json.loads((REG/'idle_launch.json').read_text())
def save(**kw):
 p=STATE/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw),indent=2)+'\n');tmp.replace(p)
def observation():
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines()
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
 assert len(gpu)==1,'Expected one GPU; revise explicit selection before launch'
 util,free=[int(x.strip()) for x in gpu[0].split(',')]
 return util==0 and not apps and free>=40960,dict(utilization=util,free_mib=free,compute_pids=apps.splitlines())
while True:
 try:idle,obs=observation()
 except Exception as e:idle=False;obs=dict(observation_error=repr(e))
 now=time.monotonic()
 if not idle:start=None
 elif start is None:start=now
 elapsed=0 if start is None else now-start
 save(phase='waiting_continuous_idle',idle_seconds=elapsed,required_seconds=1200,observation=obs)
 if elapsed>=1200:
  for path,digest in plan['hashes'].items():
   import hashlib
   if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
    save(phase='failed_source_changed',path=path);raise SystemExit(1)
  for command in plan['commands']:
   with open(command['log'],'w') as log:
    env=os.environ.copy();env['PYTHONPATH']='src:scripts'
    child=subprocess.Popen(command['argv'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
    save(phase='running',stage=command['stage'],child_pid=child.pid,log=command['log'],idle_seconds=elapsed)
    code=child.wait()
   if code:
    save(phase='failed',stage=command['stage'],returncode=code);raise SystemExit(code)
  save(phase='complete',note='Compatibility and fresh teacher collection only; H3 fitting/evaluation not yet complete.');break
 time.sleep(15)
