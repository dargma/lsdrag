"""Bounded dense indexing, serialized after the preserved small-pool collector."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from lsdrag.v11_embedding_worker import V11ProcessEmbedder
from lsdrag.embeddings import MODELS

ROOT = Path('/content/local_fast/lsdrag-text-v1')
OUT = ROOT/'corpora/musique_shared_20261023'
REPO = Path(__file__).resolve().parents[1]

def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    start = time.monotonic()
    def status(phase, **kw):
        save(OUT/'index_status.json', dict(phase=phase, pid=os.getpid(), seconds=time.monotonic()-start, **kw))
    status('waiting_collector')
    lock = (ROOT/'decision_policy.lock').open('a')
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            proc = Path('/proc/129475/cmdline')
            if not proc.exists() or b'run_decision_state_policy_iteration.py' not in proc.read_bytes():
                raise RuntimeError('GPU lock held by unexpected process; inspect')
            if time.monotonic()-start > 3600:
                raise TimeoutError('Collector wait exceeded one hour')
            time.sleep(15)
    rows = [json.loads(line) for line in (OUT/'documents.jsonl').open()]
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert sha(OUT/'documents.jsonl') == manifest['document_sha256']
    path = OUT/'vectors.partial.npy'
    assert not path.exists() and not (OUT/'vectors.npy').exists()
    vectors = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(rows), 768))
    status('loading', documents=len(rows), completed=0)
    try:
        with V11ProcessEmbedder() as embed:
            encode_start = time.monotonic()
            for first in range(0, len(rows), 128):
                if time.monotonic()-encode_start > 7200:
                    raise TimeoutError('Two-hour embedding budget')
                batch = rows[first:first+128]
                values = embed.encode([r['text'] for r in batch], 'document', titles=[r['title'] for r in batch])
                assert np.isfinite(values).all() and np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4)
                vectors[first:first+len(batch)] = values
                vectors.flush()
                done = first+len(batch)
                status('indexing', completed=done, documents=len(rows), encode_seconds=time.monotonic()-encode_start)
        del vectors
        path.rename(OUT/'vectors.npy')
        manifest.update(embeddings_complete=True, embedding_model=MODELS['gemma'], vectors_sha256=sha(OUT/'vectors.npy'))
        save(OUT/'manifest.json', manifest)
        save(REPO/'lab/registry/decision_shared_corpus_20261023/corpus_manifest.json', manifest)
        status('complete', completed=len(rows), documents=len(rows))
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise

if __name__ == '__main__':
    main()
