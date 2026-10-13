import json, subprocess
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
root=Path('/content/local_fast/lsdrag-work')
config=json.loads((root/'repo/configs/upstream.json').read_text())
for name in ['arag','m3docrag','mem_gallery']:
 info=config[name]; target=root/'upstream'/name
 if not target.exists(): subprocess.run(['git','clone',info['url'],str(target)],check=True)
 subprocess.run(['git','-C',str(target),'checkout',info['revision']],check=True)
 print('RESTORED SOURCE',name,info['revision'],flush=True)
repo='Ethan-Bei/Mem-Gallery'; rev=config['mem_gallery']['dataset_revision']
files=HfApi().list_repo_files(repo,repo_type='dataset',revision=rev)
target=root/'data/mem_gallery'
snapshot_download(repo,repo_type='dataset',revision=rev,local_dir=target,allow_patterns=['data/dialog/*.json'],max_workers=4)
(target/'source_manifest.json').write_text(json.dumps({'repo_id':repo,'revision':rev,'available_files':files},indent=2))
print('MEM GALLERY METADATA RESTORED',flush=True)
