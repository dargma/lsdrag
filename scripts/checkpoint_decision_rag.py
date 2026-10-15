"""Publish verified phase reports and one immutable Drive archive at a boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


def checkpoint(run, tag, details):
    repo=Path(__file__).resolve().parents[1]
    if not tag.replace('_','').replace('-','').isalnum():raise ValueError('Invalid tag')
    target=Path('/content/drive/MyDrive/lsdrag-backups/20261015/decision_rag_20261015')
    prior=repo/'lab/reports/decision_rag_20261015'/('checkpoint_'+tag+'.json')
    if prior.exists():
        record=json.loads(prior.read_text())
        if record['run']!=str(run) or record['details']!=details:
            raise ValueError('Existing checkpoint identity/details differ')
        if hashlib.sha256(Path(record['archive']).read_bytes()).hexdigest()!=record['archive_sha256']:
            raise ValueError('Existing Drive checkpoint changed')
        return record
    staging=run.parent.parent/'backups';staging.mkdir(parents=True,exist_ok=True)
    archive=staging/(tag+'.tar.gz')
    if archive.exists():raise FileExistsError('Immutable phase archive already exists')
    tmp=archive.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz') as tf:
        tf.add(run,arcname=run.name)
        tf.add('/content/local_fast/lsdrag-head-v12/data',arcname='data')
        tf.add(repo/'lab/registry/decision_rag_20261015',arcname='registry')
        tf.add(repo/'lab/reports/decision_rag_20261015',arcname='reports')
    tmp.replace(archive)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    target.mkdir(parents=True,exist_ok=True)
    remote=target/archive.name
    if remote.exists():raise FileExistsError('Drive archive already exists')
    shutil.copy2(archive,remote.with_suffix('.tmp'));remote.with_suffix('.tmp').replace(remote)
    if hashlib.sha256(remote.read_bytes()).hexdigest()!=digest:raise ValueError('Drive archive hash mismatch')
    record={'phase':tag,'run':str(run),'updated_utc_epoch':time.time(),'archive':str(remote),
        'archive_sha256':digest,'archive_bytes':archive.stat().st_size,'details':details,
        'controlled_latency_claims':False,'world_model':'deferred'}
    (target/(tag+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    (repo/'lab/reports/decision_rag_20261015'/('checkpoint_'+tag+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    (repo/'memory/EXPERIMENT_STATUS.json').write_text(json.dumps(record,indent=2)+'\n')
    note=f'# Phase checkpoint: {tag}\n\n'+json.dumps(record,ensure_ascii=False)+'\n\n'
    for name in ['CURRENT.md','HEAD_EXPERIMENT.md']:
        p=repo/'memory'/name;p.write_text(note+p.read_text())
    with (repo/'memory/SESSION_LOG.md').open('a') as f:f.write('\n## '+tag+'\n\n'+json.dumps(record,ensure_ascii=False)+'\n')
    subprocess.run(['git','add','memory','lab/reports/decision_rag_20261015'],cwd=repo,check=True)
    subprocess.run(['git','commit','-m','Save verified decision RAG phase '+tag],cwd=repo,check=True)
    subprocess.run(['git','push','origin','jev_v1.1'],cwd=repo,check=True)
    for name in ['CURRENT.md','HEAD_EXPERIMENT.md','EXPERIMENT_STATUS.json','SESSION_LOG.md']:
        shutil.copy2(repo/'memory'/name,Path('/content/drive/MyDrive/lsdrag/memory')/name)
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--tag',required=True);p.add_argument('--details',type=Path,required=True);a=p.parse_args()
    print(json.dumps(checkpoint(a.run,a.tag,json.loads(a.details.read_text())),ensure_ascii=False))
