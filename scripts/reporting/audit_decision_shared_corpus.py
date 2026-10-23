"""Verify full public-corpus identity and independently cached vector alignment."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/content/local_fast/lsdrag-text-v1')
REPO=Path(__file__).resolve().parents[2]
POOL=ROOT/'corpora/musique_shared_20261023'

def read(path):return json.loads(path.read_text())
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def identity(title,text):
    value=json.dumps([title,text],ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(value.encode()).hexdigest()

def main():
    m=read(POOL/'manifest.json')
    assert digest(POOL/'documents.jsonl')==m['document_sha256']
    assert digest(POOL/'vectors.npy')==m['vectors_sha256']
    rows=[json.loads(line) for line in (POOL/'documents.jsonl').open()]
    vector=np.load(POOL/'vectors.npy',mmap_mode='r')
    assert len(rows)==m['documents'] and vector.shape==(len(rows),768)
    assert all(set(r)=={'chunk_id','title','text'} for r in rows)
    positions={r['chunk_id']:i for i,r in enumerate(rows)}
    assert len(positions)==len(rows)
    hashes=[identity(r['title'],r['text']) for r in rows]
    assert hashes==sorted(hashes)
    assert all(r['chunk_id']=='c'+h[:24] for r,h in zip(rows,hashes))
    min_norm,max_norm=10.,0.
    for first in range(0,len(rows),4096):
        block=np.asarray(vector[first:first+4096]);assert np.isfinite(block).all()
        norm=np.linalg.norm(block,axis=1);assert np.allclose(norm,1,atol=1e-4)
        min_norm=min(min_norm,float(norm.min()));max_norm=max(max_norm,float(norm.max()))
    train=set(read(ROOT/'results/decision_state_policy_iteration_20261023/manifest.json')['train'])
    public=set();occurrences=0;checks=[];checked_questions=[]
    cache=ROOT/'results/text_integrated_20261016_v5_fast_only/index'
    for split in ['train','dev']:
        source=Path(m['sources'][split]['path']);assert digest(source)==m['sources'][split]['sha256']
        for line in source.open():
            question=json.loads(line)
            paragraphs=question['paragraphs']
            public.update(identity(p['title'],p['paragraph_text']) for p in paragraphs)
            occurrences+=len(paragraphs)
            path=cache/f'{question["id"]}.npy'
            if question['id'] in train and path.exists() and len(checked_questions)<8:
                old=np.load(path);assert old.shape==(len(paragraphs),768)
                for index,p in enumerate(paragraphs):
                    chunk='c'+identity(p['title'],p['paragraph_text'])[:24]
                    new=vector[positions[chunk]]
                    cosine=float(np.dot(new,old[index])/(np.linalg.norm(new)*np.linalg.norm(old[index])))
                    assert cosine>.999, (question['id'],index,cosine)
                    checks.append(cosine)
                checked_questions.append(question['id'])
    assert public==set(hashes) and occurrences==m['paragraph_occurrences']
    result=dict(audit_passed=True,documents=len(rows),paragraph_occurrences=occurrences,public_fields_only=True,all_source_paragraphs_included=True,sorted_content_id_alignment=True,vector_shape=list(vector.shape),vector_norm_range=[min_norm,max_norm],independent_cached_train_questions=checked_questions,independent_vector_pairs=len(checks),minimum_cached_vector_cosine=min(checks) if checks else None,document_sha256=m['document_sha256'],vectors_sha256=m['vectors_sha256'],note='Complete identity/finite/norm audit, with cached-vector alignment on the listed training samples only. Does not measure retrieval relevance or demonstrate complete Wikipedia coverage.')
    out=REPO/'lab/reports/decision_shared_corpus_20261023';out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
