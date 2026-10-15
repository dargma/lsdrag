#!/usr/bin/env python3
"""Add original MMQA public tables without changing previously selected questions.

All annotations and baseline proof are pinned. Writes a separate correction
manifest; never changes existing questions, corpus, gold, or their manifest.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '4dd14328c6d02a4daa357cc6032915a0b14602e3'
BASE = f'https://raw.githubusercontent.com/allenai/multimodalqa/{REVISION}/'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_jsonl(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return [json.loads(line) for line in f]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True); args.output.mkdir(parents=True, exist_ok=True)
    source_files = {}; source_paths = {}
    for rel in ['dataset/MMQA_train.jsonl.gz', 'dataset/MMQA_tables.jsonl.gz',
                'dataset/MMQA_texts.jsonl.gz', 'dataset/MMQA_images.jsonl.gz', 'baselines/pipeline.py']:
        path = args.cache / (REVISION + '_' + Path(rel).name)
        if not path.exists():
            with urllib.request.urlopen(BASE + rel, timeout=120) as response:
                path.write_bytes(response.read())
        source_paths[Path(rel).name] = path
        source_files[rel] = {'url': BASE + rel, 'sha256': sha(path)}
    questions = {q['qid']: q for q in read_jsonl(source_paths['MMQA_train.jsonl.gz'])}
    tables = {t['id']: t for t in read_jsonl(source_paths['MMQA_tables.jsonl.gz'])}
    docs = {}
    for filename in ['MMQA_texts.jsonl.gz', 'MMQA_images.jsonl.gz']:
        docs.update({d['id']: d for d in read_jsonl(source_paths[filename])})
    splits = {name: read_jsonl(args.data_dir / f'{name}_questions.jsonl')
              for name in ['train', 'development']}
    selected = splits['train'] + splits['development']
    assert len({q['qid'] for q in selected}) == len(selected), 'Duplicate qids across splits'
    backgrounds = []
    for item in selected:
        q = questions[item['qid']]
        assert item['question'] == q['question'], 'Input question differs from pinned source'
        table = tables[q['metadata']['table_id']]
        backgrounds.append({'qid': q['qid'], 'table_id': table['id'], 'title': table['title'],
            'text': json.dumps(table['table'], ensure_ascii=False, separators=(',', ':'))})
    target = args.output / 'public_backgrounds.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in backgrounds))
    def urls(items, tables_only=False):
        public = {tables[questions[q['qid']]['metadata']['table_id']]['url'] for q in items}
        if not tables_only:
            public |= {docs[s['doc_id']]['url'] for q in items
                       for s in questions[q['qid']]['supporting_context']}
        return public
    overlap = urls(splits['train']) & urls(splits['development'])
    lengths = sorted(len(r['text']) for r in backgrounds)
    stage1 = read_jsonl(args.data_dir / 'stage1_questions.jsonl')
    # Dependency components now include both gold-support and public-table URLs.
    parent = list(range(len(selected))); first = {}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, q in enumerate(selected):
        for url in urls([q]):
            if url in first: parent[root(i)] = root(first[url])
            else: first[url] = i
    components = collections.defaultdict(list)
    for i, q in enumerate(selected): components[root(i)].append(q['qid'])
    groups = {qid: hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
              for ids in components.values() for qid in ids}
    dep = args.output / 'public_dependency_groups.jsonl'
    dep.write_text(''.join(json.dumps({'qid': q['qid'], 'source_group': groups[q['qid']]}) + '\n'
                           for q in selected))
    manifest = {'revision': REVISION, 'source_files': source_files,
        'official_input_proof': BASE + 'baselines/pipeline.py#L269-L273',
        'question_selection_unchanged': True, 'existing_dataset_mutated': False,
        'input_question_hashes': {f'{n}_questions.jsonl': sha(args.data_dir / f'{n}_questions.jsonl') for n in splits},
        'serialization': 'complete original public table object; no truncation or answer-dependent selection',
        'n': len(backgrounds), 'files': {target.name: sha(target), dep.name: sha(dep)},
        'background_chars': {'p50': lengths[len(lengths)//2],
            'p95': lengths[int((len(lengths)-1)*.95)], 'max': max(lengths)},
        'stage1_table_answer_questions': sum(any(a['modality']=='table'
            for a in questions[q['qid']]['answers']) for q in stage1),
        'table_source_overlap_count': len(urls(splits['train'], True) & urls(splits['development'], True)),
        'support_plus_public_table_source_overlap_count': len(overlap),
        'overlap_urls': sorted(overlap), 'full_public_source_disjoint': not bool(overlap),
        'overlap_table_question_counts': {name: sum(tables[questions[q['qid']]['metadata']['table_id']]['url']
            in overlap for q in items) for name, items in splits.items()},
        'dependency_components': len(components)}
    (args.output / 'public_tables_correction_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
