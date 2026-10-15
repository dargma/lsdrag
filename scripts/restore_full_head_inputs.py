"""Restore only frozen full-head inputs; public annotations use direct HTTP.

Derived Mineru crops/indexes have no known exact public byte source and are read
from verified Drive backups. Never restores Qwen indexes or historical labels.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

REVISION='4dd14328c6d02a4daa357cc6032915a0b14602e3'
MMQA_URL=f'https://raw.githubusercontent.com/allenai/multimodalqa/{REVISION}/dataset/MMQA_dev.jsonl.gz'
MMQA_SHA='2e348ca574b2dc368e84671709689070f943a59f2e08e6b6e374705deb712d31'
ORIGINAL_ROOT=Path('/content/local_fast/lsdrag-work')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def public_annotations(cache):
    cache.mkdir(parents=True,exist_ok=True);path=cache/'MMQA_dev.jsonl.gz'
    if path.exists():
        if sha(path)!=MMQA_SHA:raise ValueError('Existing public annotation hash mismatch: '+str(path))
    else:
        temporary=path.with_suffix('.download-tmp')
        with urllib.request.urlopen(MMQA_URL,timeout=90) as response,temporary.open('wb') as output:
            shutil.copyfileobj(response,output)
        if sha(temporary)!=MMQA_SHA:raise ValueError('Downloaded public annotations changed')
        os.replace(temporary,path)
    return path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backup',type=Path,default=Path('/content/drive/MyDrive/lsdrag-backups/20261013'))
    p.add_argument('--target',type=Path,default=ORIGINAL_ROOT)
    p.add_argument('--cache',type=Path,default=Path('/content/local_fast/lsdrag-head-v12/source'))
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.target!=ORIGINAL_ROOT:raise ValueError('Original absolute root is required to preserve frozen index/image paths')
    if not 1<=a.workers<=16:raise ValueError('Use1–16 bounded workers')
    records={}
    for group in ['development-parsing','recovery-final','recovery','additional50-prepared','search-linked100-inputs']:
        for line in (a.backup/group/'copy_manifest.jsonl').open():
            row=json.loads(line);records[(group,row['path'])]=row
    jobs={}
    def add(group,relative):
        row=records[(group,relative)]
        if relative in jobs and jobs[relative][2]['sha256']!=row['sha256']:raise ValueError('Conflicting backup input')
        jobs[relative]=(a.backup/group/relative,a.target/relative,row)
    for group,relative in records:
        if group=='additional50-prepared' and (relative.startswith('data/ir_final288/') or
            relative.startswith('indexes_final288/gemma_') or relative.startswith('indexes_final_additions/gemma_') or
            relative in ['data/ir_final_additions/chunks.jsonl','data/ir_final_additions/figures.jsonl','data/ir_final_additions/manifest.json']):add(group,relative)
        elif group=='recovery' and relative.startswith('indexes_development/gemma_'):add(group,relative)
        elif group=='development-parsing' and relative in ['data/ir_development/chunks.jsonl','data/ir_development/figures.jsonl','data/ir_development/manifest.json']:add(group,relative)
        elif group=='search-linked100-inputs' and relative=='data/manifests/search_linked100/m3docvqa/questions.jsonl':add(group,relative)
    figures_source=a.backup/'additional50-prepared/data/ir_final288/figures.jsonl'
    expected=records[('additional50-prepared','data/ir_final288/figures.jsonl')]['sha256']
    if sha(figures_source)!=expected:raise ValueError('Figure catalog backup changed')
    figures=[json.loads(x) for x in figures_source.read_text().splitlines()]
    image_lookup={relative:group for group,relative in records if group in {'development-parsing','recovery-final'} and '/images/' in relative}
    for figure in figures:
        relative=str(Path(figure['image_path']).relative_to(a.target))
        group=image_lookup[relative]
        if records[(group,relative)]['sha256']!=figure['image_sha256']:raise ValueError('Figure/source hash mismatch')
        add(group,relative)
    plan={'files':len(jobs),'bytes':sum(j[2]['bytes'] for j in jobs.values()),'figures':len(figures),
          'target':str(a.target),'qwen_restored':False,'unreferenced_pixels_restored':False,
          'public_annotations':{'url':MMQA_URL,'sha256':MMQA_SHA},'dry_run':a.dry_run}
    print(json.dumps(plan),flush=True)
    if a.dry_run:return
    annotation=public_annotations(a.cache)
    def restore(job):
        source,target,row=job
        if target.exists():
            if sha(target)!=row['sha256']:raise ValueError('Existing target differs; refusing overwrite: '+str(target))
            state='verified_existing'
        else:
            if source.stat().st_size!=row['bytes'] or sha(source)!=row['sha256']:raise ValueError('Backup hash mismatch: '+str(source))
            target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.full-head-restore-tmp')
            shutil.copyfile(source,temporary)
            if sha(temporary)!=row['sha256']:raise ValueError('Copy hash mismatch')
            os.replace(temporary,target);state='restored'
        return {'source':str(source),'target':str(target),'bytes':row['bytes'],'sha256':row['sha256'],'status':state}
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as executor:
        results=list(executor.map(restore,jobs.values()))
    # All referenced pixels and composite part manifests are verified locally.
    for figure in figures:
        if sha(Path(figure['image_path']))!=figure['image_sha256']:raise ValueError('Final pixel validation failed')
    for channel in ['gemma_text','gemma_image']:
        manifest=json.loads((a.target/'indexes_final288'/channel/'manifest.json').read_text())
        for part in manifest['parts']:
            if sha(Path(part['directory'])/'manifest.json')!=part['manifest_sha256']:raise ValueError('Composite part changed')
    a.report.mkdir(parents=True,exist_ok=True)
    (a.report/'restore_full_head_manifest.json').write_text(json.dumps({**plan,'ready':True,
        'annotation_path':str(annotation),'files_verified':results},indent=2))
    print(json.dumps({'ready':True,'files_verified':len(results),'annotation':str(annotation)}))

if __name__=='__main__':main()
