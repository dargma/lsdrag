"""Paired pretrained D3 vs Qwen first-token selection with one non-thinking reader."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np

from lsdrag.agent import AgentConfig
from lsdrag.text_only_reader import TextOnlyReader
from lsdrag.io import sha256
from lsdrag.text_decision_benchmark import (D3Worker, canonical, public_paragraphs,
    run_episode, select_questions, TOOLS)
from lsdrag.v11_embedding_worker import V11ProcessEmbedder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/content/local_fast/lsdrag-text-v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--minutes', type=float, default=60)
    p.add_argument('--backup', type=Path)
    a = p.parse_args()
    if a.limit not in {2, 10} or not 0 < a.minutes <= 60:
        p.error('Train first2 preflight / remaining fixed dev10 paired evaluation; budget at most60min')
    a.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (a.output/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.perf_counter()
    # Every resume receives only the unspent cumulative worker budget.
    checkpoint = a.output/'status.json'
    spent = json.loads(checkpoint.read_text()).get('cumulative_worker_seconds', 0) if checkpoint.exists() else 0
    deadline = started + a.minutes*60 - spent

    def save(name, value):
        out = a.output/name; out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name+'.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); os.replace(tmp, out)

    def status(phase, **details):
        save('status.json', {'phase': phase, 'pid': os.getpid(), 'updated_epoch': time.time(),
             'cumulative_worker_seconds': spent+time.perf_counter()-started, **details})

    def backup():
        if a.backup:
            a.backup.mkdir(parents=True, exist_ok=True)
            archive = a.output.parent/(a.output.name+'.tar.gz')
            with tarfile.open(archive, 'w:gz') as t:
                t.add(a.output, arcname=a.output.name)
            destination = a.backup/archive.name
            shutil.copyfile(archive, destination)
            if sha256(destination) != sha256(archive):
                raise RuntimeError('Drive archive hash mismatch')

    repo = Path(__file__).resolve().parents[1]
    src = a.root/'source/musique'
    raw = {split: [json.loads(x) for x in (src/f'musique_ans_v1.0_{split}.jsonl').read_text().splitlines()]
           for split in ('train', 'dev')}
    train, dev = select_questions(raw['train'], 48), select_questions(raw['dev'], 12)
    train_parts = {d['id'] for q in train for d in q['question_decomposition']}
    dev_parts = {d['id'] for q in dev for d in q['question_decomposition']}
    if train_parts & dev_parts:
        raise ValueError('Train/dev question decomposition family overlap')
    model = a.root/'hf/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554'
    d3 = a.root/'hf/models--vllm-sr--d3-flash/snapshots/8b35c853b90e777595ec36bacae820fe08ea077a'
    prompt = (repo/'prompts/text_rag_integrated_v2.txt').read_text()
    config = AgentConfig(max_output_tokens=512, context_limit=16384, max_images=0,
                         image_delivery_policy='agent-read')
    manifest = {'experiment': 'd3-qwen-nonthinking-text-rag-v2', 'model': str(model), 'd3': str(d3),
        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'tools': TOOLS,
        'data_sha256': {split: sha256(src/f'musique_ans_v1.0_{split}.jsonl') for split in raw},
        'train_qids': [r['id'] for r in train], 'dev_qids': [r['id'] for r in dev], 'evaluated_qids': [r['id'] for r in (train[:2] if a.limit==2 else dev[2:12])], 'partition': 'train_preflight' if a.limit==2 else 'remaining_dev10',
        'source_sha256': {name: sha256(repo/name) for name in ['scripts/benchmark_d3_qwen_text_rag.py','src/lsdrag/text_only_reader.py',
            'src/lsdrag/text_decision_benchmark.py','src/lsdrag/hf_reader.py','src/lsdrag/direct_reader.py','src/lsdrag/tool_registry.py']},
        'max_tool_calls': 8, 'max_output_tokens': 512, 'context_limit': 16384,
        'shared_gpu': True, 'controlled_latency_claims': False, 'policies': ['first_token', 'd3'],
        'data_scope': 'MuSiQue-Ans 2hop; official per-question candidate paragraphs behind tools',
        'source_provenance': json.loads((a.root/'source/musique_sources.json').read_text()),
        'versions': {k: importlib.metadata.version(k) for k in ['torch','transformers','sentence-transformers']}}
    prior = a.output/'manifest.json'
    if prior.exists() and json.loads(prior.read_text()) != manifest:
        raise ValueError('Source/data/model contract changed; use new output')
    save('manifest.json', manifest)
    sys.path.insert(0, str(a.root/'source/musique_scorer'))
    from metrics.answer import compute_exact, compute_f1
    from metrics.support import SupportMetric
    # Source metric integrity is captured before loading either GPU model.
    save('scorer_provenance.json', {str(f.relative_to(a.root/'source/musique_scorer')): sha256(f)
         for f in (a.root/'source/musique_scorer').rglob('*.py')})
    save('launch.json', {'pid': os.getpid(), 'limit': a.limit, 'argv': sys.argv,
                         'started_epoch': time.time(), 'deadline_seconds_remaining': deadline-time.perf_counter()})
    embedder, controller = None, None
    try:
        status('loading')
        free = float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'],text=True).splitlines()[0])/1024
        if free < 40:
            raise RuntimeError('Need40GiB free for the two-model functional comparison; no unrelated job is stopped')
        embedder = V11ProcessEmbedder()
        corpus_cache = {}
        for q in (train[:2] if a.limit==2 else dev[2:12]):
            if time.perf_counter() >= deadline:
                raise TimeoutError('Budget reached during document indexing')
            rows = public_paragraphs(q)
            vectors = embedder.encode([r['text'] for r in rows], 'document', titles=[r['title'] for r in rows])
            corpus_cache[q['id']] = rows, vectors
        reader = TextOnlyReader(str(model), config)
        controller = D3Worker(a.root/'d3-env/bin/python', d3)
        save('d3_runtime.json', controller.ready)
        import torch
        # Equal functional warmup; time observed only, no speed claim.
        reader.choose([{'role': 'user', 'content': 'Warmup.'}], [{'action':'FINAL','arguments':None}, {'action':'semantic_search','arguments':None}])
        controller.choose([{'role': 'user', 'content': 'Warmup.'}],
                          [{'action':'FINAL','arguments':None}, {'action':'semantic_search','arguments':None}])
        results = []
        for number, q in enumerate(train[:2] if a.limit==2 else dev[2:12]):
            schedule = ['first_token', 'd3']; random.Random(20261016+number).shuffle(schedule)
            rows, vectors = corpus_cache[q['id']]
            for policy in schedule:
                rel = Path('episodes')/policy/(q['id']+'.json')
                if (a.output/rel).exists():
                    result = json.loads((a.output/rel).read_text())
                    if result['qid'] != q['id'] or result['policy'] != policy:
                        raise ValueError('Resume identity mismatch')
                else:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('Worker budget reached')
                    result = run_episode(q['question'], rows, vectors, embedder, reader, prompt,
                                         policy, controller, deadline)
                    aliases = [q['answer']]+q['answer_aliases']
                    result.update(qid=q['id'], em=max(compute_exact(x,result['answer']) for x in aliases),
                                  f1=max(compute_f1(x,result['answer']) for x in aliases))
                    gold_ids = {f'p{p["idx"]}' for p in q['paragraphs'] if p['is_supporting']}
                    retrieved = {r['chunk_id'] for t in result['trajectory'] if t['tool_name'].endswith('search') and isinstance(t['result'],list)
                                 for r in t['result']}
                    read = {r['chunk_id'] for t in result['trajectory'] if t['tool_name']=='read_chunk' and isinstance(t['result'],list) for r in t['result']}
                    id_map = {f'p{p["idx"]}': p['idx'] for p in q['paragraphs']}
                    unknown = sorted(set(result['support_ids'])-id_map.keys())
                    unknown_map = {k: -1-i for i,k in enumerate(unknown)}
                    predicted = [id_map[k] if k in id_map else unknown_map[k] for k in result['support_ids']]
                    support = SupportMetric(); support(predicted, [id_map[k] for k in gold_ids])
                    result['unknown_support_ids'] = unknown
                    result.update(support_f1=support.get_metric()[1], retrieved_support_recall=len(gold_ids & retrieved)/len(gold_ids),
                                  read_support_recall=len(gold_ids & read)/len(gold_ids))
                    save(rel, result)
                results.append(result)
                status('evaluating', completed_episodes=len(results), qid=q['id'], policy=policy)
                print(json.dumps({'qid':q['id'],'policy':policy,'f1':result['f1'],'tool_calls':result['tool_calls'],'errors':result['errors']}),flush=True)
            backup()
        summary = {}
        for policy in ['first_token','d3']:
            rows = [r for r in results if r['policy']==policy]
            summary[policy] = {'n':len(rows), **{key:float(np.mean([r[key] for r in rows])) for key in
                ['f1','em','support_f1','retrieved_support_recall','read_support_recall','seconds','tool_calls',
                 'reused_argument_calls','argument_tokens','prompt_tokens','completion_tokens','answer_format_valid']},
                'episodes_with_errors':sum(bool(r['errors']) for r in rows)}
        save('summary.json', {'policies':summary,'controlled_latency_claims':False,'trained_head':False,
            'scope':'Small paired 2hop development subset; external controller baseline; not full Jev/MuSiQue benchmark',
            'comparison_note':'Same reader, execution candidates and argument/final generator; no trained heads. Different decision backbone/prompt serialization; two-model D3 cost included.',
            'peak_main_allocated_bytes':torch.cuda.max_memory_allocated()})
        status('complete', completed_episodes=len(results)); backup()
    except TimeoutError as error:
        status('budget_limited', error=str(error)); backup()
    except Exception as error:
        status('failed', error=type(error).__name__+': '+str(error)); backup(); raise
    finally:
        if controller: controller.close()
        if embedder: embedder.close()


if __name__ == '__main__':
    main()
