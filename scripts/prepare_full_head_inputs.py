"""Reproduce exact outcome-blind full-head100/20 inputs from frozen sources.

Writes only new/byte-identical files, never overwrites changed active inputs.
--dry-run builds temporary artifacts and checks ordered qids/file hashes.
"""
from pathlib import Path
import argparse
import collections
import gzip
import hashlib
import json
import random
import tempfile

from restore_full_head_inputs import public_annotations, sha


def load(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def put(path,text):
    payload=text.encode()
    if path.exists():
        if path.read_bytes()!=payload:raise ValueError('Existing file differs; refusing overwrite: '+str(path))
    else:path.write_bytes(payload)


def prepare(a,out):
    out.mkdir(parents=True,exist_ok=True)
    source=public_annotations(a.cache)
    allq=[json.loads(x) for x in gzip.open(source,'rt')]
    docpath=a.corpus/'documents.jsonl';docs=load(docpath)
    mapping={i:d for d in docs for i in d.get('document_ids',[d['document_id']])}
    qs=[q for q in allq if {s['doc_id'] for s in q['supporting_context']}<=mapping.keys() and q['metadata']['table_id'] in mapping]
    oldpath=a.prior_questions;old={q['qid'] for q in load(oldpath)}
    parents=list(range(len(qs)));first={}
    def root(i):
     while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
     return i
    def urls(q):return {mapping[s['doc_id']]['url'] for s in q['supporting_context']}|{mapping[q['metadata']['table_id']]['url']}
    for i,q in enumerate(qs):
     for url in urls(q):
      if url in first:parents[root(i)]=root(first[url])
      else:first[url]=i
    groups=collections.defaultdict(list)
    for i,q in enumerate(qs):groups[root(i)].append(q)
    components=list(groups.values());components.sort(key=lambda g:sorted(q['qid'] for q in g));rng=random.Random(20261016);rng.shuffle(components)
    devpool=[];devcomp=[]
    for g in components:
     if not any(q['qid'] in old for q in g) and len(devpool)<20:devpool+=g;devcomp.append(g)
    assert len(devpool)>=20
    trainpool=[q for g in components if not any(g is d for d in devcomp) for q in g];rng.shuffle(trainpool);rng.shuffle(devpool)
    splits={'train':trainpool[:100],'development':devpool[:20]};assert not set.union(*(urls(q) for q in splits['train']))&set.union(*(urls(q) for q in splits['development']))
    sourcegroups={q['qid']:hashlib.sha256('\n'.join(sorted(url for x in g for url in urls(x))).encode()).hexdigest() for g in components for q in g}
    for name,qset in splits.items():
     put(out/f'{name}_questions.jsonl',''.join(json.dumps({'qid':q['qid'],'question':q['question']})+'\n' for q in qset))
     put(out/f'{name}_gold_private.jsonl',''.join(json.dumps(q)+'\n' for q in qset))
     put(out/f'{name}_gold.jsonl',''.join(json.dumps({'qid':q['qid'],'answers':[str(a['answer']) for a in q['answers']],'source_group':sourcegroups[q['qid']], 'support_document_ids':sorted({mapping[s['doc_id']]['document_id'] for s in q['supporting_context']}|{mapping[q['metadata']['table_id']]['document_id']})})+'\n' for q in qset))
    put(out/'dev_questions.jsonl',(out/'development_questions.jsonl').read_text())
    put(out/'gold.jsonl',(out/'train_gold.jsonl').read_text()+(out/'development_gold.jsonl').read_text())
    put(out/'eligible_questions.jsonl',''.join(json.dumps({'qid':q['qid'],'question':q['question']})+'\n' for q in qs))
    put(out/'source_groups.jsonl',''.join(json.dumps({'qid':q['qid'],'source_group':sourcegroups[q['qid']],'source_urls':sorted(urls(q))})+'\n' for q in qs))
    expected_hashes={
        'train_questions.jsonl':'1c91e0797e0699b3de6d9ed240a9c984aa26408cb6ee7ba802c094671e1ed79e',
        'dev_questions.jsonl':'62a6895c897a9a4c2950c9f557a8474c935ae178f07ab22f7d413de15040c76d',
        'gold.jsonl':'ceb20b6d7d7ebca04945e47e7cba7454c05832623d3b26aa9cacd3e79f581de7',
        'source_groups.jsonl':'ea7ea50ebaa76934c071c0d196ecd712290c10a3d662a39f8a53e8c52cc38757'}
    for filename,digest in expected_hashes.items():
        if sha(out/filename)!=digest:raise ValueError('Frozen input reproduction mismatch: '+filename)
    if len(qs)!=124 or len(splits['train'])!=100 or len(splits['development'])!=20:
        raise ValueError('Frozen eligibility/split counts changed')
    files={f.name:sha(f) for f in out.glob('*.jsonl')}
    if a.compare_to:
        for filename,digest in files.items():
            if sha(a.compare_to/filename)!=digest:raise ValueError('Current/reference file differs: '+filename)
    manifest={'seed':20261016,'source':'official MMQA dev annotations, same question lineage as M3DocVQA',
        'source_sha256':sha(source),'question_selection':'outcome-blind eligibility: all support and public table aliases covered by fixed final288 corpus; no corpus alteration',
        'eligible':len(qs),'components':len(components),'train_prefixes':[60,80,100],
        'ordered_qids':{n:[q['qid'] for q in items] for n,items in splits.items()},
        'train':100,'development':20,'source_support_plus_table_overlap':0,
        'prior100_overlap':{n:sum(q['qid'] in old for q in items) for n,items in splits.items()},
        'development_selection':'components with no prior100 question qids preferred; fixedseed only, no outcomes',
        'not_a_final_test':True,'no_unseen_source_claim':'source supports between new splits disjoint; corpus shared and previously used; retrieval exposure audit still required',
        'no_old_predictions_or_policy_labels_loaded':True,'prior100_question_hash':sha(oldpath),'files':files,
        'corpus_path':str(a.corpus),'indexes_path':str(a.indexes),
        'reproduction_sources':{'documents_sha256':sha(docpath),'script_sha256':sha(Path(__file__)),
            'public_annotations':str(source),'prior_questions':str(oldpath)},
        'manifest_identity_note':'Reproduction provenance differs from the original frozen manifest. Do not replace the registered active manifest; retain that original identity separately.'}
    put(out/'reproduced_manifest.json',json.dumps(manifest,indent=2))
    return {k:manifest[k] for k in ['eligible','components','train','development','prior100_overlap','files']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,default=Path('/content/local_fast/lsdrag-work/data/ir_final288'))
    p.add_argument('--indexes',type=Path,default=Path('/content/local_fast/lsdrag-work/indexes_final288'))
    p.add_argument('--prior-questions',type=Path,default=Path('/content/local_fast/lsdrag-work/data/manifests/search_linked100/m3docvqa/questions.jsonl'))
    p.add_argument('--cache',type=Path,default=Path('/content/local_fast/lsdrag-head-v12/source'))
    p.add_argument('--output',type=Path)
    p.add_argument('--compare-to',type=Path)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not a.dry_run and a.output is None:p.error('--output is required unless --dry-run')
    if a.output and (a.output/'manifest.json').exists():
        raise ValueError('Directory has an active/frozen manifest; use a new output directory or --dry-run')
    if a.dry_run:
        with tempfile.TemporaryDirectory(prefix='full-head-input-check-',dir='/content/local_fast') as temporary:
            result=prepare(a,Path(temporary))
    else:result=prepare(a,a.output)
    print(json.dumps({'dry_run':a.dry_run,'reproduced_exact_frozen_inputs':True,**result},indent=2))

if __name__=='__main__':main()
