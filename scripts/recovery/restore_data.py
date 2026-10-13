import concurrent.futures, hashlib, json, shutil, time
from pathlib import Path
root=Path('/content/local_fast/lsdrag-work')
backup=Path('/content/drive/MyDrive/lsdrag-backups/20261013')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def copy(item):
 src,dst,expected=item
 dst.parent.mkdir(parents=True,exist_ok=True)
 before=expected or digest(src)
 if not dst.exists() or digest(dst)!=before:
  tmp=dst.with_name(dst.name+'.restore-tmp'); shutil.copyfile(src,tmp)
  if digest(tmp)!=before: raise ValueError(str(src))
  tmp.replace(dst)
 return {'path':str(dst.relative_to(root)),'sha256':before,'bytes':dst.stat().st_size}
items=[(p,root/'data'/p.relative_to(backup/'preflight'),None) for p in (backup/'preflight').rglob('*') if p.is_file()]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: records=list(pool.map(copy,items))
print('preflight verified',len(records),flush=True)
rows=[json.loads(l) for l in (backup/'development-parsing/copy_manifest.jsonl').read_text().splitlines()]
items=[(backup/'development-parsing'/r['path'],root/r['path'],r['sha256']) for r in rows if r['status']!='failed']
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for i,r in enumerate(pool.map(copy,items),1):
  records.append(r)
  if i%250==0: print('development verified',i,flush=True)
(root/'recovery/restored_files.json').write_text(json.dumps(records,indent=2))
print('RESTORE COMPLETE',len(records),sum(r['bytes'] for r in records),flush=True)
