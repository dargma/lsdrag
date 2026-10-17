"""Frozen action-head versus full native, outcome-blind new 2-hop100."""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lsdrag.text_contraction_probe import Budget
from lsdrag.text_decision_benchmark import TOOLS, public_paragraphs
from lsdrag.text_integrated import run_episode, selection_messages
from lsdrag.text_integrated_aligned import AlignedReader

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[1]
ORIGIN = ROOT/'results/improvement_v3_merged_20261017'
REGISTRY = REPO/'lab/registry/text_integrated_20261016/generalization100_manifest_20261017.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sources():
    return {name: sha(REPO/name) for name in ['scripts/run_frozen_head_generalization100.py',
            'src/lsdrag/text_integrated.py', 'src/lsdrag/text_integrated_aligned.py',
            'src/lsdrag/text_contraction_probe.py', 'src/lsdrag/text_only_reader.py',
            'src/lsdrag/text_decision_benchmark.py']}


def prepare():
    if REGISTRY.exists():
        raise FileExistsError('Never overwrite a frozen question manifest')
    pattern = re.compile(r'\b[234]hop__[0-9_]+\b')
    exposed, audit = set(), {}
    # Conservative available-artifact audit: includes planned/reserved IDs.
    # Do not scan public raw dataset files or unrelated repositories.
    directories = [REPO/'lab', REPO/'memory', REPO/'docs', ROOT/'results']
    for directory in directories:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix in ['.json', '.jsonl', '.md', '.csv']:
                found = set(pattern.findall(path.read_text(errors='replace')))
                found.update(pattern.findall(path.name))
                if found:
                    exposed.update(found)
                    audit[str(path)] = {'sha256': sha(path), 'qids_found': len(found)}
    raw = {}
    for name in ['train', 'dev']:
        with (ROOT/f'source/musique/musique_ans_v1.0_{name}.jsonl').open() as stream:
            raw[name] = [json.loads(line) for line in stream]
    families = {d['id'] for q in raw['train']+raw['dev'] if q['id'] in exposed for d in q['question_decomposition']}
    population = [q for q in raw['dev'] if q['answerable'] and len(q['question_decomposition']) == 2]
    eligible = [q for q in population if q['id'] not in exposed and not {d['id'] for d in q['question_decomposition']}&families]
    ordered = sorted(eligible, key=lambda q: hashlib.sha256(('generalization100_20261017:'+q['id']).encode()).hexdigest())
    selected, used, skipped = [], set(), 0
    for q in ordered:
        family = {d['id'] for d in q['question_decomposition']}
        if family & used:
            skipped += 1
            continue
        selected.append(q['id']); used.update(family)
        if len(selected) == 100:
            break
    assert len(selected) == 100 and not set(selected)&exposed and not used&families
    original = json.loads((ORIGIN/'frozen_primary.json').read_text())
    assert sha(ORIGIN/'head.pt') == original['head_sha256']
    assert sha(REPO/'prompts/text_rag_integrated_v5.txt') == original['prompt_sha256']
    for name, digest in original['source_sha256'].items():
        assert sha(REPO/name) == digest, name
    manifest = {'profile': 'text_integrated_20261016/frozen_head_generalization100_20261017',
                'qids': selected, 'excluded_available_exposed_qids': sorted(exposed),
                'excluded_decomposition_family_count': len(families),
                'selection': 'SHA256(generalization100_20261017:qid), then greedy disjoint atomic question families among selected questions',
                'population_2hop_answerable_dev': len(population), 'eligible_after_prior_exclusions': len(eligible),
                'within_new_set_family_overlap_skipped_before100': skipped,
                'audit_sources': audit, 'known_exposure_limit': 'Available local results and Git lab/memory/docs, including reserved IDs. Public pretraining contamination cannot be excluded.',
                'model': original['model'], 'head_sha256': original['head_sha256'],
                'prompt_sha256': original['prompt_sha256'], 'source_sha256': sources(),
                'dataset_sha256': {name: sha(ROOT/f'source/musique/musique_ans_v1.0_{name}.jsonl') for name in ['train', 'dev']},
                'policies': ['native', 'head_generate'], 'episodes': 200,
                'order': 'paired, even question native/head; odd head/native',
                'max_output_tokens': 512, 'context_limit': 16384, 'tool_cap': 8,
                'worker_seconds_excluding_load': 1800, 'generated_tokens_cap': 150000,
                'qwen_prefill_plus_forced_prefix_cap': 3000000,
                'training': False, 'read_selection': False, 'decoding': 'greedy non-thinking',
                'primary_metrics': ['paired_answer_f1', 'paired_answer_em'],
                'secondary': ['support_f1', 'tool_calls', 'interaction_turns', 'errors', 'token_cost', 'observed_episode_latency'],
                'quality_early_stopping': False, 'retuning_on_this100': False,
                'claims': 'larger frozen-policy evaluation; no formal noninferiority or controlled shared-GPU speed claim',
                'next': 'report all100; no automatic retraining/3hop expansion'}
    REGISTRY.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ['population_2hop_answerable_dev', 'eligible_after_prior_exclusions',
                                             'excluded_decomposition_family_count', 'within_new_set_family_overlap_skipped_before100']}, indent=2))


class MeteredReader(AlignedReader):
    budget = None

    def capture(self, messages, options, mode, deadline):
        self.budget.reserve(inputs=int(self.prepare(selection_messages(messages, options), TOOLS)['input_ids'].shape[1]), generated=0)
        result = super().capture(messages, options, mode, deadline)
        self.budget.charge(result[2])
        return result

    def generate_native(self, messages, trajectory, deadline=float('inf')):
        self.budget.reserve(inputs=int(self.prepare(messages, TOOLS)['input_ids'].shape[1]))
        result = super().generate_native(messages, trajectory, deadline)
        self.budget.charge(result[1])
        return result

    def continue_cached(self, action, trajectory, deadline=float('inf')):
        self.budget.reserve(inputs=64)
        result = super().continue_cached(action, trajectory, deadline)
        self.budget.charge(result[1])
        return result

    def generate_schema(self, messages, schema, mode, instruction, deadline=float('inf')):
        self.budget.reserve(inputs=int(self.prepare(messages, TOOLS)['input_ids'].shape[1])+512)
        result = super().generate_schema(messages, schema, mode, instruction, deadline)
        self.budget.charge(result[1])
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.output is None:
        parser.error('--output required')
    manifest = json.loads(REGISTRY.read_text())
    assert sources() == manifest['source_sha256']
    assert sha(ORIGIN/'head.pt') == manifest['head_sha256']
    assert sha(REPO/'prompts/text_rag_integrated_v5.txt') == manifest['prompt_sha256']
    assert all(sha(ROOT/f'source/musique/musique_ans_v1.0_{name}.jsonl') == digest for name, digest in manifest['dataset_sha256'].items())
    args.output.mkdir(parents=True, exist_ok=False)
    import fcntl
    import torch
    from lsdrag.v11_embedding_worker import V11ProcessEmbedder
    lock = (ROOT/'improvement_v3.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def save(name, value):
        path = args.output/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    started = time.perf_counter(); budget = None; episodes = []
    def phase(value, **extra):
        state = {'phase': value, 'pid': os.getpid(), 'completed': len(episodes), 'elapsed_seconds': time.perf_counter()-started, **extra}
        if budget:
            state.update(worker_seconds=1800-(budget.deadline-time.perf_counter()), generated_tokens=budget.generated, input_tokens=budget.inputs)
        save('status.json', state); print(value, state, flush=True)
    save('manifest.json', manifest)
    questions = {}
    with (ROOT/'source/musique/musique_ans_v1.0_dev.jsonl').open() as stream:
        for line in stream:
            q = json.loads(line)
            if q['id'] in manifest['qids']:
                questions[q['id']] = q
    assert len(questions) == 100
    sys.path.insert(0, str(ROOT/'source/musique_scorer'))
    from metrics.answer import compute_exact, compute_f1
    embedder = None
    try:
        phase('loading')
        embedder = V11ProcessEmbedder()
        reader = MeteredReader(manifest['model'], SimpleNamespace(max_output_tokens=512, context_limit=16384))
        ck = torch.load(ORIGIN/'head.pt', map_location='cpu', weights_only=True)
        assert not set(manifest['qids']) & set(ck['train_qids'])
        reader.checkpoint = {k: ck[k].to('cuda') for k in ['action_weight', 'action_bias', 'slot_weight', 'slot_bias']}
        reader.budget = budget = Budget(seconds=1800, generated=150000, inputs=3000000)
        prompt = (REPO/'prompts/text_rag_integrated_v5.txt').read_text()
        for index, qid in enumerate(manifest['qids']):
            q = questions[qid]; rows = public_paragraphs(q)
            budget.reserve(generated=0)
            tick = time.perf_counter()
            vectors = embedder.encode([r['text'] for r in rows], 'document', titles=[r['title'] for r in rows])
            save(f'corpus/{qid}.json', {'seconds': time.perf_counter()-tick, 'paragraphs': len(rows),
                                     'vectors_sha256': hashlib.sha256(vectors.tobytes()).hexdigest()})
            np.save(args.output/f'corpus/{qid}.npy', vectors)
            order = manifest['policies'] if index % 2 == 0 else list(reversed(manifest['policies']))
            for policy in order:
                phase('evaluation', question_index=index, qid=qid, policy=policy)
                ep, _, _ = run_episode(q['question'], rows, vectors, embedder, reader, prompt,
                                       'native' if policy == 'native' else 'head', 'always_generate', budget.deadline)
                gold = {f'p{p["idx"]}' for p in q['paragraphs'] if p['is_supporting']}; pred = set(ep['support_ids'])
                ep.update(qid=qid, policy=policy,
                          f1=max(compute_f1(a, ep['answer']) for a in [q['answer']]+q['answer_aliases']),
                          em=max(compute_exact(a, ep['answer']) for a in [q['answer']]+q['answer_aliases']),
                          support_f1=2*len(gold & pred)/(len(gold)+len(pred)))
                ep['interaction_turns'] = sum(c['kind'] in ['native_combined', 'decision'] for c in ep['reader_calls'])+int(ep['forced_final'])
                ep['forced_prefix_tokens'] = sum(c['usage'].get('forced_prefix_tokens', 0) for c in ep['reader_calls'])
                ep['prefill_count'] = sum(c['usage'].get('full_prefill_count', 0) for c in ep['reader_calls'])
                ep['decoder_forward_count_derived'] = sum(max(0, c['usage'].get('generated_tokens', 0)-1)+int(c['usage'].get('forced_prefix_tokens', 0)>0)
                                                        for c in ep['reader_calls'] if c['usage'].get('generated_tokens', 0)>0)
                assert ep['reused_argument_calls'] == 0
                save(f'episodes/{policy}/{qid}.json', ep); episodes.append(ep)
                print('result', qid, policy, ep['f1'], ep['errors'], flush=True)
        phase('complete', pid=None, training=False)
    except BaseException as error:
        phase('failed_or_budget_limited', pid=None, error=repr(error)); raise
    finally:
        if embedder:
            embedder.close()


if __name__ == '__main__':
    main()
