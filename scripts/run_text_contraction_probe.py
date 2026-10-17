"""Frozen train24/validation8 counterfactual probe, with hard resource caps."""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lsdrag.text_contraction_probe import Budget, ProbeReader, answer_after_first_tool, continuation
from lsdrag.text_decision_benchmark import TOOLS, canonical, public_paragraphs
from lsdrag.text_integrated import ACTIONS, observed_ids, options_for

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[1]
TEACHER = ROOT/'results/text_integrated_20261016_v5_fast_only'
ORIGIN = ROOT/'results/improvement_v3_merged_20261017'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_inputs():
    split = json.loads((REPO/'lab/registry/text_integrated_20261016/improvement_v2_split.json').read_text())
    states = []
    for group in ['train', 'validation']:
        for qid in split[group]:
            path = TEACHER/'teacher'/f'{qid}.json'
            d = json.loads(path.read_text())
            r = next(r for r in d['records'] if len(r['history']) == 1)
            states.append({'group': group, 'qid': qid, 'loop': r['loop'],
                           'source_sha256': sha(path), 'source': str(path),
                           'messages_sha256': hashlib.sha256(canonical(r['messages']).encode()).hexdigest(),
                           'history_sha256': hashlib.sha256(canonical(r['history']).encode()).hexdigest()})
    assert len(states) == 32 and len({s['qid'] for s in states}) == 32
    frozen = json.loads((ORIGIN/'frozen_primary.json').read_text())
    assert sha(ORIGIN/'head.pt') == frozen['head_sha256']
    assert sha(REPO/'prompts/text_rag_integrated_v5.txt') == frozen['prompt_sha256']
    for name, digest in frozen['source_sha256'].items():
        assert sha(REPO/name) == digest, name
    return {'profile': 'text_integrated_20261016/contraction_probe_20261017',
            'states': states, 'head_sha256': frozen['head_sha256'], 'model': frozen['model'],
            'prompt_sha256': frozen['prompt_sha256'],
            'source_sha256': {n: sha(REPO/n) for n in ['src/lsdrag/text_contraction_probe.py',
                                                      'scripts/run_text_contraction_probe.py',
                                                      'src/lsdrag/text_integrated.py',
                                                      'src/lsdrag/text_integrated_aligned.py']},
            'caps': {'worker_seconds_excluding_load': 1800, 'generated_tokens': 120000,
                     'qwen_prefill_plus_forced_prefix_tokens': 2000000,
                     'base_continuations': 160, 'path_checks': 16, 'B1_answers': 64,
                     'total_probes': 240, 'suffix_tool_calls': 2,
                     'max_output_tokens': 512, 'context_limit': 16384},
            'selection': 'one state after first real tool per existing train24/validation8; no outcome selection',
            'smoke': 'first8 train questions; native vs aligned path checks and first8 continuations',
            'branches': ['head', 'native_name_boundary_common_aligned', 'FINAL', 'up to2 other legal tools'],
            'native_selector': 'same original prefill, independent cloned KV for greedy name-prefix decode; all routing tokens/time charged; original cache retained for common aligned args; duplicated routing syntax is diagnostic overhead, not optimized native latency',
            'alternates': 'keyword,semantic,read priority, exclude head first action; observed-ID legality only',
            'budget_B1': 'separate FINAL generation after first actual suffix tool for head/native baseline',
            'no_future_suffix': True, 'training': False, 'oldtest_or_newtest_used': False,
            'gate': {'correct_train_min': 5, 'preserved_quality_shorter_questions_min': 3,
                     'nonterminal_shortcuts_min': 3, 'calls_or_turns_reduction': .20,
                     'preserve_correct_em_and_per_question_f1_and_observed_support_recall': True},
            'validation': 'already exposed confirmation; branch/gate frozen before train; no retuning',
            'timing': 'shared GPU observation only; saved historical prefix cost separate, suffix measured now; corpus prep/load/scoring outside suffix',
            'original_primary_unchanged': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    registry = REPO/'lab/registry/text_integrated_20261016/contraction_probe_manifest_20261017.json'
    if args.prepare:
        if registry.exists():
            raise FileExistsError('Frozen manifest already exists')
        registry.write_text(json.dumps(frozen_inputs(), ensure_ascii=False, indent=2)+'\n')
        print(registry)
        return
    if args.output is None:
        parser.error('--output required for execution')
    manifest = json.loads(registry.read_text())
    assert manifest == frozen_inputs(), 'Frozen input/code identity changed'
    args.output.mkdir(parents=True, exist_ok=False)
    import fcntl
    import torch
    from lsdrag.v11_embedding_worker import V11ProcessEmbedder
    lock = (ROOT/'improvement_v3.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def save(name, value):
        path = args.output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    started = time.perf_counter()
    budget = None
    completed, checked = [], []
    def phase(value, **extra):
        state = {'phase': value, 'pid': os.getpid(), 'elapsed_seconds': time.perf_counter()-started,
                 'completed': len(completed), 'path_checks': len(checked), **extra}
        if budget:
            state.update(actual_generated_tokens=budget.generated, actual_input_tokens=budget.inputs,
                         worker_seconds=1800-(budget.deadline-time.perf_counter()))
        save('status.json', state)
        print(value, state, flush=True)
    save('manifest.json', manifest)
    raw = {}
    with (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open() as stream:
        for line in stream:
            q = json.loads(line)
            if q['id'] in {s['qid'] for s in manifest['states']}:
                raw[q['id']] = q
    assert len(raw) == 32
    sys.path.insert(0, str(ROOT/'source/musique_scorer'))
    from metrics.answer import compute_exact, compute_f1
    def score(ep, q):
        gold = {f'p{p["idx"]}' for p in q['paragraphs'] if p['is_supporting']}
        read = {r['chunk_id'] for t in ep['trajectory'] if t['tool_name'] == 'read_chunk' for r in t['result']}
        seen = set(observed_ids(ep['trajectory']))
        pred = set(ep['support_ids'])
        ep.update(f1=max(compute_f1(a, ep['answer']) for a in [q['answer']]+q['answer_aliases']),
                  em=max(compute_exact(a, ep['answer']) for a in [q['answer']]+q['answer_aliases']),
                  observed_support_recall=len(gold & seen)/len(gold),
                  read_support_recall=len(gold & read)/len(gold),
                  citation_observed_valid=pred <= seen,
                  support_f1=2*len(pred & gold)/(len(pred)+len(gold)))
        usage = [c['usage'] for c in ep['reader_calls']]
        ep.update(generated_tokens=sum(u.get('generated_tokens', u.get('completion_tokens', 0)) for u in usage),
                  argument_tokens=sum(c['usage'].get('generated_tokens', 0) for c in ep['reader_calls'] if c['kind'] == 'arguments'),
                  forced_prefix_tokens=sum(u.get('forced_prefix_tokens', 0) for u in usage),
                  prefill_input_tokens=sum(u.get('prompt_tokens', 0) for u in usage if u.get('full_prefill_count', 0)),
                  full_prefill_count=sum(u.get('full_prefill_count', 0) for u in usage),
                  full_decision_turns=1+ep['decision_turns'],
                  tool_seconds=sum(t['seconds'] for t in ep['trajectory'][1:]))
        return ep
    embedder = None
    try:
        phase('loading')
        embedder = V11ProcessEmbedder()
        reader = ProbeReader(manifest['model'], SimpleNamespace(max_output_tokens=512, context_limit=16384))
        ck = torch.load(ORIGIN/'head.pt', map_location='cpu', weights_only=True)
        reader.checkpoint = {k: ck[k].to('cuda') for k in ['action_weight', 'action_bias', 'slot_weight', 'slot_bias']}
        budget = Budget()
        for index, state in enumerate(manifest['states']):
            qid, group = state['qid'], state['group']
            phase('functional_smoke' if index < 8 else group+'_probe', qid=qid)
            source = json.loads(Path(state['source']).read_text())
            record = next(r for r in source['records'] if r['loop'] == state['loop'])
            messages, history = record['messages'], record['history']
            q = raw[qid]
            rows = public_paragraphs(q)
            vectors = np.load(TEACHER/'index'/f'{qid}.npy')
            if index < 8:
                # Full native vs the same chosen action under the common cache
                # generator; values/results must match, no free routing claim.
                inputs = int(reader.prepare(messages, TOOLS)['input_ids'].shape[1])
                budget.reserve(inputs=inputs)
                value, usage, error = reader.generate_native(messages, history, budget.deadline)
                budget.charge(usage)
                if error:
                    raise RuntimeError('Path smoke native format failure: '+error)
                action = value['action']
                budget.reserve(inputs=inputs, generated=0)
                _, _, capture = reader.capture(messages, options_for(q['question'], history), 'head', budget.deadline)
                budget.charge(capture)
                budget.reserve()
                routed, route_usage, route_error = reader.native_action_cached(budget.deadline)
                budget.charge(route_usage)
                budget.reserve(inputs=64)
                aligned, au, ae = reader.continue_cached(action, history, budget.deadline)
                budget.charge(au)
                expected = value.get('final') if action == 'FINAL' else value['arguments']
                check = {'qid': qid, 'native_action': action, 'routed_action': routed,
                         'route_matches': routed == action and route_error is None,
                         'values_match': aligned == expected and ae is None,
                         'native_value': expected, 'aligned_value': aligned,
                         'native_usage': usage, 'route_usage': route_usage,
                         'capture_usage': capture, 'aligned_usage': au}
                checked.append(check)
                save(f'path_checks/{qid}.json', check)
                if not check['route_matches'] or not check['values_match']:
                    raise RuntimeError('Path functional gate failed: '+qid)
            head = None
            legal = [a for a in ACTIONS[:-1] if any(o['action'] == a for o in options_for(q['question'], history))]
            initial = ['head', 'native', 'FINAL']
            for policy in initial:
                ep, snapshot = continuation(q['question'], rows, vectors, embedder, reader, messages, history, policy, budget)
                ep = score(ep, q)
                ep.update(qid=qid, group=group, branch=policy,
                          prefix_sha256=state['history_sha256'],
                          historical_prefix_tool_seconds=history[0]['seconds'])
                save(f'episodes/{group}/{qid}/{policy}.json', ep)
                completed.append({'qid': qid, 'group': group, 'branch': policy})
                print('result', qid, policy, ep['f1'], ep['suffix_tool_calls'], flush=True)
                if policy == 'head':
                    head = ep['actions'][0]['executed_action'] if ep['actions'] else None
                if snapshot and policy in ['head', 'native']:
                    one = score(answer_after_first_tool(reader, snapshot, budget), q)
                    one.update(qid=qid, group=group, branch=policy+'_B1')
                    save(f'episodes/{group}/{qid}/{policy}_B1.json', one)
                    completed.append({'qid': qid, 'group': group, 'branch': policy+'_B1'})
            for action in [a for a in legal if a != head][:2]:
                ep, _ = continuation(q['question'], rows, vectors, embedder, reader, messages, history, action, budget)
                ep = score(ep, q)
                ep.update(qid=qid, group=group, branch='alt_'+action)
                save(f'episodes/{group}/{qid}/alt_{action}.json', ep)
                completed.append({'qid': qid, 'group': group, 'branch': 'alt_'+action})
            if index == 7:
                save('functional_gate.json', {'passed': True, 'questions': 8, 'path_checks': checked,
                                             'cache_isolation': 'recompute each branch; native route clones cache and checks original length',
                                             'not_quality_generalization': True})
            assert len(completed)+2*len(checked) <= 240
        phase('complete', pid=None, training=False)
    except BaseException as error:
        phase('failed_or_budget_limited', error=repr(error), pid=None)
        raise
    finally:
        if embedder:
            embedder.close()


if __name__ == '__main__':
    main()
