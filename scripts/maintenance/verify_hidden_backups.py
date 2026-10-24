"""Verify cached hidden tensors against Drive archive members, in two CPU workers."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,tarfile,time
ROOT=Path('/content/local_fast/lsdrag-text-v1/results')
DRIVE=Path('/content/drive/MyDrive/lsdrag-backups/20261023')
OUT=Path('lab/registry/hidden_backup_20261024')
def digest(f):
 h=hashlib.sha256()
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def check(p):
 with p.open('rb') as f: expected=digest(f)
 archive=DRIVE/(p.parent.name+'.tar.gz')
 if p.parent.name=='decision_policy_20261023_pilot1':archive=DRIVE/'decision_policy_pilot1_results.tar.gz'
 found=None
 if archive.exists():
  with tarfile.open(archive,'r:gz') as t:
   matches=[m for m in t.getmembers() if m.isfile() and (m.name==p.name or m.name.endswith('/'+p.name))]
   for m in matches:
    if m.size==p.stat().st_size:
     with t.extractfile(m) as f: actual=digest(f)
     if actual==expected:found=m.name;break
 if found is None:
  # Exact derived bytes have no public replacement. Preserve source as a standalone backup.
  import shutil
  folder=DRIVE/'hidden_cache_20261024'/p.parent.name;folder.mkdir(parents=True,exist_ok=True)
  dst=folder/p.name
  if dst.exists():
   with dst.open('rb') as f:assert digest(f)==expected,'Existing backup mismatch'
  else:
   temp=dst.with_suffix(dst.suffix+'.partial');shutil.copy2(p,temp)
   with temp.open('rb') as f:assert digest(f)==expected
   temp.rename(dst)
  archive=dst;found=None
 return dict(source=str(p),bytes=p.stat().st_size,sha256=expected,backup=str(archive),archive_member=found,verified=True)
def main():
 paths=sorted(p for p in ROOT.rglob('*.pt') if p.name in ('states.pt','teacher_states.pt','train_features.pt','validation_features.pt'))
 OUT.mkdir(parents=True,exist_ok=True)
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(check,paths))
 report=dict(verified_epoch=time.time(),files=results,count=len(results),total_bytes=sum(r['bytes'] for r in results),scope='Explicit extracted state/teacher/feature caches under text-v1/results; rollout batch embeddings are covered by their separate run archives.')
 (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='files'}))
if __name__=='__main__':main()
