#!/usr/bin/env python3
"""Prepare outcome-blind source-disjoint MMQA pilot and a shared reduced corpus.

Official annotations are used only to construct the shared corpus, split sources,
and private scoring files. They never enter public question inputs. This reduced
support-preserving corpus is not the original open-domain MMQA benchmark.
"""
import argparse, collections, concurrent.futures, gzip, hashlib, io, json
import pathlib, random, struct, urllib.request, zipfile, zlib
BASE = 'https://raw.githubusercontent.com/allenai/multimodalqa/master/dataset/'
ZIP = 'https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip'

def get(url, start=None, end=None):
    headers = {} if start is None else {'Range': f'bytes={start}-{end}'}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as r:
        if start is not None and r.status != 206:
            raise RuntimeError('Server did not honor byte range')
        return r.read()

def rows(p):
    with gzip.open(p, 'rt') as f:
        return [json.loads(x) for x in f]

def write_rows(p, rs):
    p.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rs))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cache', type=pathlib.Path, required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True); ap.add_argument('--seed', type=int, default=20261015)
    ap.add_argument('--train', type=int, default=300); ap.add_argument('--development', type=int, default=80)
    ap.add_argument('--distractors', type=int, default=200); args=ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True); args.output.mkdir(parents=True, exist_ok=True)
    rng=random.Random(args.seed)
    for n in ['train','texts','images']:
        p=args.cache/f'MMQA_{n}.jsonl.gz'
        if not p.exists(): p.write_bytes(get(BASE+p.name))
    texts=rows(args.cache/'MMQA_texts.jsonl.gz'); images=rows(args.cache/'MMQA_images.jsonl.gz')
    docs={x['id']:x for x in texts+images}
    qs=[q for q in rows(args.cache/'MMQA_train.jsonl.gz') if set(q['metadata']['modalities'])=={'text','image'}
        and 'TableQ' not in q['metadata']['type']
        and {s['doc_part'] for s in q['supporting_context']}=={'text','image'}]
    parent=list(range(len(qs))); first={}
    def root(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    for i,q in enumerate(qs):
        for s in q['supporting_context']:
            key=docs[s['doc_id']]['url']
            if key in first: parent[root(i)]=root(first[key])
            else: first[key]=i
    groups=collections.defaultdict(list)
    for i,q in enumerate(qs): groups[root(i)].append(q)
    group_by_qid={q['qid']:hashlib.sha256('\n'.join(sorted(x['qid'] for x in component)).encode()).hexdigest() for component in groups.values() for q in component}
    components=list(groups.values()); rng.shuffle(components)
    dev_pool=[]; train_pool=[]
    for component in components:
        (dev_pool if len(dev_pool)<args.development else train_pool).extend(component)
    rng.shuffle(dev_pool); rng.shuffle(train_pool)
    splits={'train':train_pool[:args.train], 'development':dev_pool[:args.development]}
    assert all(len(splits[n])==getattr(args,n) for n in splits)
    def sources(qset): return {docs[s['doc_id']]['url'] for q in qset for s in q['supporting_context']}
    assert not sources(splits['train']) & sources(splits['development'])
    chosen=sum(splits.values(),[]); ids={s['doc_id'] for q in chosen for s in q['supporting_context']}
    # Corpus is shared across every question, including all distractors. Not gold per-query.
    corpus={}
    for modality,pool in [('text',texts),('image',images)]:
        support=[x for x in pool if x['id'] in ids]
        extra=[x for x in pool if x['id'] not in ids]; rng.shuffle(extra)
        corpus[modality]=support+extra[:args.distractors]
    imgdir=args.output/'images'; imgdir.mkdir(exist_ok=True)
    # Read only ZIP central directory then selected compressed payloads.
    with urllib.request.urlopen(urllib.request.Request(ZIP,method='HEAD'), timeout=90) as r:
        size=int(r.headers['Content-Length'])
    tail=get(ZIP,max(0,size-65536),size-1); ix=tail.rfind(b'PK\x05\x06')
    eocd=tail[ix:ix+22]; fields=struct.unpack('<4s4H2LH',eocd); cdsize,cdoffset=fields[5:7]
    cd=get(ZIP,cdoffset,cdoffset+cdsize-1)
    z=zipfile.ZipFile(io.BytesIO(cd+eocd)); infos={pathlib.PurePosixPath(x.filename).name:x for x in z.infolist()}
    def fetch_image(item):
        info=infos[pathlib.PurePosixPath(item['path']).name]; dest=imgdir/pathlib.PurePosixPath(item['path']).name
        if not dest.exists():
            off=info.header_offset+cdoffset; head=get(ZIP,off,off+29)
            h=struct.unpack('<4s5H3L2H',head); start=off+30+h[-2]+h[-1]
            data=get(ZIP,start,start+info.compress_size-1)
            if info.compress_type==8: data=zlib.decompress(data,-15)
            elif info.compress_type!=0: raise RuntimeError('Unsupported ZIP compression')
            if zlib.crc32(data)&0xffffffff != info.CRC: raise RuntimeError('ZIP CRC mismatch')
            dest.write_bytes(data)
        return {'id':item['id'],'modality':'image','path':str(dest.resolve()),
                'title':item['title'],'url':item['url'],'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        imagerows=list(executor.map(fetch_image,corpus['image']))
    write_rows(args.output/'corpus_images.jsonl',imagerows)
    write_rows(args.output/'corpus_texts.jsonl',[dict(x,modality='text') for x in corpus['text']])
    for name,qset in splits.items():
        write_rows(args.output/f'{name}_questions.jsonl',[{'qid':q['qid'],'question':q['question']} for q in qset])
        write_rows(args.output/f'{name}_gold_private.jsonl',[{'qid':q['qid'],'answers':q['answers'],
            'supporting_context':q['supporting_context'],'question_type':q['metadata']['type']} for q in qset])
    # Balanced by declared composition type before examining any model output.
    buckets=collections.defaultdict(list)
    for q in splits['train']: buckets[q['metadata']['type']].append(q)
    smoke=[]
    while len(smoke)<40:
        for key in sorted(buckets):
            if buckets[key] and len(smoke)<40: smoke.append(buckets[key].pop())
    write_rows(args.output/'stage1_questions.jsonl',[{'qid':q['qid'],'question':q['question']} for q in smoke])
    write_rows(args.output/'stage1_gold_private.jsonl',[{'qid':q['qid'],'answers':q['answers'],
        'supporting_context':q['supporting_context'],'question_type':q['metadata']['type']} for q in smoke])
    generic_corpus=[{'item_id':x['id'],'modality':'text','text':x['text'],'title':x['title']} for x in corpus['text']]
    generic_corpus += [{'item_id':x['id'],'modality':'image','text':x['title'],'title':x['title'],'image_path':x['path']} for x in imagerows]
    write_rows(args.output/'corpus.jsonl',generic_corpus)
    gold=[{'qid':q['qid'],'answers':[a['answer'] for a in q['answers']],
           'support_ids':[s['doc_id'] for s in q['supporting_context']],
           'source_group':group_by_qid[q['qid']],
           'question_type':q['metadata']['type']} for q in chosen]
    write_rows(args.output/'gold.jsonl',gold)
    manifest={'seed':args.seed,'dataset':'official allenai/MultiModalQA train release',
      'source_url':BASE,'images_url':ZIP,'question_counts':{k:len(v) for k,v in splits.items()},
      'stage1_is_train_subset':True,'stage1_n':len(smoke),'eligible_questions':len(qs),
      'source_components':len(groups),'split_unit':'connected components of supporting document Wikipedia URLs',
      'support_source_overlap':0,'entity_disjoint_claim':False,'candidate_context_disjoint_claim':False,
      'prior_poc_qid_exclusion':'not performed: new MMQA source; prior M3DocVQA can share semantic questions',
      'corpus_counts':{k:len(v) for k,v in corpus.items()},
      'corpus_policy':'One shared support-preserving reduced corpus: union of selected gold supports plus 200 random text and 200 random image distractors. Not original full-corpus benchmark.',
      'license':'Official repository lacks dataset LICENSE; Wikipedia text and Wikimedia images retain source-specific licenses. Record attribution URLs; do not claim a blanket dataset license.',
      'files':{str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob('*.jsonl')}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
