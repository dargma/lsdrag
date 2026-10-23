"""Build a question-independent public paragraph pool; never select by gold labels."""
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[1]
OUT = ROOT / 'corpora/musique_shared_20261023'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    documents = {}
    sources = {}
    occurrences = 0
    for split in ['train', 'dev']:
        path = ROOT / f'source/musique/musique_ans_v1.0_{split}.jsonl'
        sources[split] = dict(path=str(path), sha256=sha(path))
        for line in path.open():
            question = json.loads(line)
            # Only public document fields. Do not read answer/support/decomposition.
            for paragraph in question['paragraphs']:
                title, text = paragraph['title'], paragraph['paragraph_text']
                identity = json.dumps([title, text], ensure_ascii=False, separators=(',', ':'))
                digest = hashlib.sha256(identity.encode()).hexdigest()
                row = dict(chunk_id='c'+digest[:24], title=title, text=text)
                if digest in documents:
                    assert documents[digest] == row
                documents[digest] = row
                occurrences += 1
    ordered = [documents[k] for k in sorted(documents)]
    assert len({r['chunk_id'] for r in ordered}) == len(ordered)
    corpus = OUT / 'documents.jsonl'
    with corpus.open('w') as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    db = OUT / 'lexical.sqlite'
    with sqlite3.connect(db) as con:
        con.execute('CREATE VIRTUAL TABLE documents USING fts5(chunk_id UNINDEXED, title, text)')
        con.executemany('INSERT INTO documents VALUES (:chunk_id,:title,:text)', ordered)
    manifest = dict(
        corpus_name=OUT.name, documents=len(ordered), paragraph_occurrences=occurrences,
        sources=sources, document_fields=['chunk_id', 'title', 'text'],
        document_sha256=sha(corpus), lexical_sha256=sha(db), path=str(OUT),
        selection='All public paragraphs in official train and dev sources; exact title+text deduplication; no query-specific pools or gold-based inclusion.',
        evaluation_note='Public evaluation document text is searchable, as in open-corpus retrieval. Evaluation questions/answers/support labels are not passed to the corpus builder output or training targets. This is a dataset-derived shared corpus, not full Wikipedia.',
        embeddings_complete=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    reg = REPO / 'lab/registry/decision_shared_corpus_20261023'
    reg.mkdir(parents=True, exist_ok=True)
    (reg/'corpus_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
