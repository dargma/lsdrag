"""Frozen v3 read-argument selection ON/OFF on the previously evaluated20.

Post-hoc exploratory evaluation; never changes the original primary or checkpoint.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lsdrag.text_integrated import run_episode
from lsdrag.text_integrated_aligned import AlignedReader
from lsdrag.text_decision_benchmark import public_paragraphs
from lsdrag.v11_embedding_worker import V11ProcessEmbedder


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import fcntl
    import torch
    root = Path('/content/local_fast/lsdrag-text-v1')
    repo = Path(__file__).resolve().parents[1]
    origin = root / 'results/improvement_v3_merged_20261017'
    teacher = root / 'results/text_integrated_20261016_v5_fast_only'
    lock = (root / 'improvement_v3.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = json.loads((origin / 'frozen_primary.json').read_text())
    assert sha(origin / 'head.pt') == frozen['head_sha256']
    assert sha(repo / 'prompts/text_rag_integrated_v5.txt') == frozen['prompt_sha256']
    for name, digest in frozen['source_sha256'].items():
        assert sha(repo / name) == digest, name
    checkpoint = torch.load(origin / 'head.pt', map_location='cpu', weights_only=True)
    assert checkpoint['format'] == 'text-action-read-v3'
    assert checkpoint['search_slots'] == 'GENERATE_ONLY'
    qids = frozen['qids']
    assert len(qids) == len(set(qids)) == 20
    assert not set(qids) & set(checkpoint['train_qids'])
    manifest = {
        'profile': 'text_integrated_20261016/v3_argument_selection_posthoc',
        'qids': qids, 'head_sha256': sha(origin / 'head.pt'),
        'original_frozen_primary_sha256': sha(origin / 'frozen_primary.json'),
        'prompt_sha256': frozen['prompt_sha256'], 'model': frozen['model'],
        'source_sha256': {**frozen['source_sha256'],
                          'scripts/evaluate_v3_argument_selection.py': sha(Path(__file__))},
        'policies': ['always_generate', 'read_selection'],
        'order': 'paired alternating OFF/ON, then ON/OFF by question index',
        'training': 'none; original checkpoint unchanged',
        'scope': 'read arguments: GENERATE or existing observed candidates1..4; search always generates',
        'evaluation_status': 'post-hoc: qids and original outcomes previously observed; no fresh-heldout claim',
        'max_output_tokens': 512, 'context_limit': 16384, 'worker_minutes': 30,
        'timing': 'episode wall incl search/query embedding; excludes load/corpus precompute/scoring/writes; uncontrolled observational timing',
        'no_retuning': True, 'original_primary_unchanged': True,
    }
    def save(name, value):
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    start = time.perf_counter()
    deadline = start + 1800
    def phase(value, **extra):
        save('status.json', {'phase': value, 'pid': os.getpid(),
                            'worker_seconds': time.perf_counter() - start, **extra})
        print(value, extra, flush=True)
    save('manifest.json', manifest)
    phase('loading')
    questions = {}
    with (root / 'source/musique/musique_ans_v1.0_dev.jsonl').open() as stream:
        for line in stream:
            q = json.loads(line)
            if q['id'] in qids:
                questions[q['id']] = q
    assert len(questions) == 20
    sys.path.insert(0, str(root / 'source/musique_scorer'))
    from metrics.answer import compute_exact, compute_f1
    embedder = None
    episodes = []
    try:
        embedder = V11ProcessEmbedder()
        reader = AlignedReader(frozen['model'], SimpleNamespace(max_output_tokens=512, context_limit=16384))
        reader.checkpoint = {k: checkpoint[k].to('cuda') for k in
                             ['action_weight', 'action_bias', 'slot_weight', 'slot_bias']}
        prompt = (repo / 'prompts/text_rag_integrated_v5.txt').read_text()
        for index, qid in enumerate(qids):
            q = questions[qid]
            rows = public_paragraphs(q)
            vector_path = teacher / 'index' / f'{qid}.npy'
            vectors = np.load(vector_path) if vector_path.exists() else embedder.encode(
                [row['text'] for row in rows], 'document', titles=[row['title'] for row in rows])
            order = ['always_generate', 'read_selection']
            if index % 2:
                order.reverse()
            for policy in order:
                phase('evaluation', completed=len(episodes), qid=qid, policy=policy)
                ep, _, _ = run_episode(q['question'], rows, vectors, embedder, reader, prompt,
                                       'head', 'reuse' if policy == 'read_selection' else 'always_generate', deadline)
                aliases = [q['answer']] + q['answer_aliases']
                gold = {f'p{p["idx"]}' for p in q['paragraphs'] if p['is_supporting']}
                pred = set(ep['support_ids'])
                ep.update(qid=qid, policy=policy,
                          f1=max(compute_f1(ans, ep['answer']) for ans in aliases),
                          em=max(compute_exact(ans, ep['answer']) for ans in aliases),
                          support_f1=2*len(gold & pred)/(len(gold)+len(pred)) if gold or pred else 0.)
                ep['interaction_turns'] = sum(c['kind'] == 'decision' for c in ep['reader_calls']) + int(ep['forced_final'])
                ep['read_calls'] = sum(t['tool_name'] == 'read_chunk' for t in ep['trajectory'])
                ep['selected_read_calls'] = sum(t['tool_name'] == 'read_chunk' and t['argument_reused'] for t in ep['trajectory'])
                ep['generated_read_calls'] = ep['read_calls'] - ep['selected_read_calls']
                ep['forced_prefix_tokens'] = sum(c['usage'].get('forced_prefix_tokens', 0) for c in ep['reader_calls'])
                ep['prefill_count'] = sum(c['usage'].get('full_prefill_count', 0) for c in ep['reader_calls'])
                assert ep['selected_read_calls'] == ep['reused_argument_calls']
                if policy == 'always_generate':
                    assert ep['selected_read_calls'] == 0
                save(f'evaluation/{policy}/{qid}.json', ep)
                episodes.append(ep)
                print('result', qid, policy, ep['f1'], ep['selected_read_calls'], ep['errors'], flush=True)
        keys = ['f1', 'em', 'support_f1', 'tool_calls', 'interaction_turns', 'read_calls',
                'selected_read_calls', 'generated_read_calls', 'argument_generation_calls',
                'argument_tokens', 'completion_tokens', 'prompt_tokens', 'forced_prefix_tokens',
                'prefill_count', 'seconds']
        policies = {}
        for policy in manifest['policies']:
            es = [ep for ep in episodes if ep['policy'] == policy]
            policies[policy] = {'n': len(es), **{k: float(np.mean([ep[k] for ep in es])) for k in keys},
                                'errors': sum(bool(ep['errors']) for ep in es),
                                'selected_read_total': sum(ep['selected_read_calls'] for ep in es),
                                'read_total': sum(ep['read_calls'] for ep in es),
                                'latency_total_seconds': sum(ep['seconds'] for ep in es)}
        pairs = []
        for qid in qids:
            es = {ep['policy']: ep for ep in episodes if ep['qid'] == qid}
            off, on = es['always_generate'], es['read_selection']
            prior = json.loads((origin / 'evaluation/always_generate' / f'{qid}.json').read_text())
            pairs.append({'qid': qid, 'f1_off': off['f1'], 'f1_on': on['f1'],
                          'f1_delta': on['f1']-off['f1'], 'answer_off': off['answer'], 'answer_on': on['answer'],
                          'selected_reads_on': on['selected_read_calls'],
                          'off_matches_original_answer': off['answer'] == prior['answer'],
                          'off_matches_original_trajectory': off['trajectory'] and
                          [(t['tool_name'], t['arguments'], t['result']) for t in off['trajectory']] ==
                          [(t['tool_name'], t['arguments'], t['result']) for t in prior['trajectory']]})
        save('summary.json', {'policies': policies, 'pairs': pairs,
                             'wins': sum(p['f1_delta'] > 0 for p in pairs),
                             'ties': sum(p['f1_delta'] == 0 for p in pairs),
                             'losses': sum(p['f1_delta'] < 0 for p in pairs),
                             'exploratory_posthoc': True, 'controlled_latency_claim': False,
                             'original_primary_unchanged': True})
        phase('complete', completed=40)
    except BaseException as error:
        phase('failed_or_budget_limited', completed=len(episodes), error=repr(error))
        raise
    finally:
        if embedder:
            embedder.close()


if __name__ == '__main__':
    main()
