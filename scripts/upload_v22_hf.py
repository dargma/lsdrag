"""Back up finished v2.2 runs to private Hugging Face repos and verify remote sha256.

results: each run directory becomes one tar.gz of its small files (episodes, json); collection group
tensors go to a separate uncompressed tar; model checkpoints (model/ + head.pt) go to a model repo as-is.
Optimizer resume states are never uploaded. Usage:
  upload_v22_hf.py results RUN [RUN ...]      -> dataset sungkwang2/lsdrag-v22-generated, claude_runs/<date>/
  upload_v22_hf.py model RUN NAME             -> model   sungkwang2/lsdrag-v22-checkpoints, NAME/
"""
import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path('/content/local_fast/lsdrag-text-v1')
STAGE = ROOT/'exports/v22_claude_hf_20261110'
DATASET = 'sungkwang2/lsdrag-v22-generated'
MODELS = 'sungkwang2/lsdrag-v22-checkpoints'
PREFIX = 'claude_runs/20261110'
SKIP = {'resume.pt', 'resume.pt.tmp'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 26), b''):
            h.update(block)
    return h.hexdigest()


def remote_sha(api, repo, kind, paths):
    info = api.get_paths_info(repo, paths, repo_type=kind, expand=True)
    return {i.path: (i.lfs.sha256 if getattr(i, 'lfs', None) else None) for i in info}


def stage_run(name):
    run = ROOT/'results'/name
    assert (run/'status.json').exists(), run
    assert json.loads((run/'status.json').read_text())['phase'] in ('complete', 'failed', 'gate_failed'), name
    STAGE.mkdir(parents=True, exist_ok=True)
    out = {}
    small = STAGE/f'{name}.tar.gz'
    with tarfile.open(small, 'w:gz') as tar:
        for p in sorted(run.rglob('*')):
            rel = p.relative_to(run)
            if p.is_file() and not p.is_symlink() and rel.parts[0] not in ('model', 'groups') and p.name not in SKIP:
                tar.add(p, arcname=f'{name}/{rel}')
    out[small] = f'{PREFIX}/results/{small.name}'
    if (run/'groups').is_dir():
        big = STAGE/f'{name}_groups.tar'
        with tarfile.open(big, 'w') as tar:
            tar.add(run/'groups', arcname=f'{name}/groups')
        out[big] = f'{PREFIX}/groups/{big.name}'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['results', 'model'])
    ap.add_argument('args', nargs='+')
    a = ap.parse_args()
    api = HfApi()
    record = STAGE/'uploads.json'
    STAGE.mkdir(parents=True, exist_ok=True)
    log = json.loads(record.read_text()) if record.exists() else []
    if a.kind == 'results':
        files = {}
        for name in a.args:
            files.update(stage_run(name))
        for local, remote in files.items():
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=DATASET, repo_type='dataset',
                            commit_message=f'Claude v2.2 backup: {remote}')
        got = remote_sha(api, DATASET, 'dataset', list(files.values()))
        for local, remote in files.items():
            digest = sha(local)
            assert got[remote] == digest, (remote, got[remote], digest)
            log.append(dict(repo=DATASET, path=remote, sha256=digest, bytes=local.stat().st_size,
                            verified_utc=datetime.now(timezone.utc).isoformat()))
            local.unlink()
    else:
        name, target = a.args
        run = ROOT/'results'/name
        assert json.loads((run/'status.json').read_text())['phase'] == 'complete'
        api.create_repo(MODELS, repo_type='model', private=True, exist_ok=True)
        assert api.repo_info(MODELS, repo_type='model').private
        files = {p: f'{target}/{p.relative_to(run)}' for p in sorted(run.rglob('*'))
                 if p.is_file() and not p.is_symlink() and p.name not in SKIP and p.relative_to(run).parts[0] != 'groups'}
        api.upload_folder(folder_path=str(run), path_in_repo=target, repo_id=MODELS, repo_type='model',
                          ignore_patterns=['resume.pt*', 'groups/*'], commit_message=f'Claude v2.2 checkpoint: {target} from {name}')
        lfs = {r: l for l, r in files.items() if l.suffix in ('.safetensors', '.pt')}
        got = remote_sha(api, MODELS, 'model', list(lfs))
        for remote, local in lfs.items():
            digest = sha(local)
            assert got[remote] == digest, (remote, got[remote], digest)
            log.append(dict(repo=MODELS, path=remote, sha256=digest, bytes=local.stat().st_size,
                            verified_utc=datetime.now(timezone.utc).isoformat()))
    record.write_text(json.dumps(log, indent=2)+'\n')
    print(json.dumps(log[-len(a.args):] if a.kind == 'results' else log[-3:], indent=2))


if __name__ == '__main__':
    main()
