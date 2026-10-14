"""Prepare only frozen teacher-training inputs; holdout is never read."""
import argparse
import gzip
import json
from pathlib import Path
from lsdrag.io import read_jsonl,write_jsonl,dump_json,sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work-root',type=Path,required=True)
    a=p.parse_args();r=a.work_root
    repo=Path(__file__).resolve().parents[1]
    plan=json.loads((repo/'lab/registry/jev_v1.0/label_collection_plan.json').read_text())
    docs=read_jsonl(r/'data/ir_final288/documents.jsonl')
    figures=read_jsonl(r/'data/ir_final288/figures.jsonl')
    doc_by_url={d['url']:d for d in docs}
    figure_docs={f['document_id'] for f in figures if Path(f['image_path']).is_file() and sha256(f['image_path'])==f['image_sha256']}
    meta={x['id']:x for name in ['MMQA_texts','MMQA_tables','MMQA_images'] for x in read_jsonl(r/'data/mmqa'/(name+'.jsonl'))}
    source=r/'recovery/jev_v1.0/mmqa_train.jsonl.gz'
    assert sha256(source)==plan['mmqa_train_url_covered_candidates']['source_sha256']
    candidates=set(plan['mmqa_train_url_covered_candidates']['qids'])
    accepted=[];qc=[]
    for line in gzip.open(source,'rt'):
        q=json.loads(line)
        if q['qid'] not in candidates: continue
        missing=[]
        for c in q['supporting_context']:
            d=doc_by_url.get(meta[c['doc_id']]['url'])
            if d is None or (c['doc_part']=='image' and d['document_id'] not in figure_docs): missing.append(c)
        qc.append({'qid':q['qid'],'passed':not missing,'missing':missing})
        if not missing: accepted.append(q)
    dest=r/'data/manifests/jev_v1.0/mmqa_train_qc'
    accepted.sort(key=lambda q:q['qid'])
    write_jsonl(dest/'questions.jsonl',[{'qid':q['qid'],'question':q['question']} for q in accepted])
    write_jsonl(dest/'gold.jsonl',accepted)
    dump_json(dest/'qc.json',{'rows':qc,'accepted':len(accepted),'meaning':'URL coverage and image file/hash availability only; not verified gold-crop alignment. No original corpus aliases or indexes changed.'})
    print(f'MMQA accepted {len(accepted)} / {len(candidates)}',flush=True)
    # Sole fallback: entire public news corpus; never evidence_list-only corpus.
    src=r/'recovery/jev_v1.0';fallback=plan['fallback']
    assert sha256(src/'MultiHopRAG.json')==fallback['questions_sha256']
    assert sha256(src/'corpus.json')==fallback['corpus_sha256']
    articles=json.loads((src/'corpus.json').read_text());questions=json.loads((src/'MultiHopRAG.json').read_text())
    corpus=r/'data/jev_multihop_corpus';chunks=[];documents=[]
    for i,d in enumerate(articles):
        did=f'mh{i:04d}'
        documents.append({'document_id':did,'document_ids':[did],**{k:d.get(k) for k in ['title','url','source','published_at']}})
        body=d['body'];parts=[body[s:s+1600] for s in range(0,len(body),1600)]
        for j,part in enumerate(parts):
            cid=f'{did}:c{j:04d}'
            chunks.append({'chunk_id':cid,'document_id':did,'document_ids':[did],'page_index':0,'kind':'text','title':d['title'],'text':f"Title: {d['title']}\nSource: {d['source']}\nPublished: {d['published_at']}\n{part}",'figure_ids':[],'previous_chunk_id':f'{did}:c{j-1:04d}' if j else None,'next_chunk_id':f'{did}:c{j+1:04d}' if j+1<len(parts) else None})
    write_jsonl(corpus/'documents.jsonl',documents);write_jsonl(corpus/'chunks.jsonl',chunks);write_jsonl(corpus/'figures.jsonl',[])
    dump_json(corpus/'manifest.json',{'dataset':'yixuantt/MultiHopRAG','documents':len(documents),'chunks':len(chunks),'source_sha256':sha256(src/'corpus.json'),'chunking':'1600 characters without overlap; complete bodies plus public document metadata','chunks_sha256':sha256(corpus/'chunks.jsonl')})
    ordered=[];gold=[]
    for idx in fallback['source_rows_in_execution_order']:
        q=questions[idx];qid=f'multihop_{idx:04d}'
        ordered.append({'qid':qid,'question':q['query']})
        gold.append({'qid':qid,'question':q['query'],'answers':[{'answer':q['answer']}],'answerable':q['question_type']!='null_query','question_type':q['question_type']})
    dest=r/'data/manifests/jev_v1.0/multihop_train1000'
    write_jsonl(dest/'questions.jsonl',ordered);write_jsonl(dest/'gold.jsonl',gold)
    print(f'MultiHop frozen {len(ordered)} inputs; full corpus {len(documents)} docs / {len(chunks)} chunks',flush=True)

if __name__=='__main__': main()
