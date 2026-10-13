import json, subprocess, time
from pathlib import Path
r=Path('/content/local_fast/lsdrag-work'); repo=r/'repo'
rows=[json.loads(line) for line in (r/'data/manifests/final288/render_new.jsonl').read_text().splitlines()]
pdfs=r/'data/pdfs_final_additions'; parsed=r/'data/parsed_final_additions'
deadline=time.monotonic()+14400
while not all((pdfs/(row['document_id']+'.json')).exists() for row in rows):
 if time.monotonic()>deadline: raise TimeoutError('Final PDF collection did not finish')
 time.sleep(15)
missing=[row['document_id'] for row in rows if not (pdfs/(row['document_id']+'.pdf')).exists()]
if missing: raise RuntimeError('Missing PDFs: '+str(missing))
subprocess.run([str(r/'parser-env/bin/python'),str(repo/'scripts/parse_pdfs.py'),'--input',str(pdfs),'--output',str(parsed),'--tier','standard','--resume'],cwd=repo,check=True)
missing=[row['document_id'] for row in rows if not (parsed/row['document_id']/'_SUCCESS.json').exists()]
if missing: raise RuntimeError('Failed parses: '+str(missing))
subprocess.run([str(r/'runtime/bin/python'),str(repo/'scripts/backup_local.py'),'--source-root',str(r),'--destination','/content/drive/MyDrive/lsdrag-backups/20261013/recovery-final','--include','data/pdfs_final_additions','--include','data/parsed_final_additions','--include','data/manifests/final288','--report',str(r/'recovery/final-backup.json')],cwd=repo,check=True)
print('FINAL DATA RESTORED AND BACKED UP',flush=True)
