"""Post-collection equivalence/timing probe; never runs a learned policy.

Gold support selects diagnostic inputs only. No answers or policy labels enter
prompts. This script must run alone on the GPU after the frozen collection.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from lsdrag.io import dump_json, read_jsonl, sha256
from lsdrag.v11_backend import V11Backend
from lsdrag.v11_fast_readout import fast_encode_state
from lsdrag.v11_protocol import DECISION, input_hash, observation_messages


def diagnostic_inputs(data, limit):
    items = {r['item_id']: r for r in read_jsonl(data/'corpus.jsonl')}
    gold = {r['qid']: r for r in read_jsonl(data/'gold.jsonl')}
    backgrounds_path = data/'public_backgrounds.jsonl'
    backgrounds = {r['qid']: r for r in read_jsonl(backgrounds_path)} if backgrounds_path.exists() else {}
    samples = []
    for q in read_jsonl(data/'train_questions.jsonl')[:limit]:
        evidence = []
        background = backgrounds.get(q['qid'])
        if background:
            evidence.append({'item_id': 'public_table:'+background['table_id'], 'modality': 'text',
                             'title': background.get('title', 'Public question table'),
                             'text': background['text'], 'public_table': True})
        support = [items[k] for k in dict.fromkeys(gold[q['qid']]['support_ids'])]
        evidence.extend(support)
        messages = observation_messages(q['question'], evidence, DECISION)
        samples.append({'qid': q['qid'], 'messages': messages, 'input_hash': input_hash(messages),
                        'modalities': sorted({r['modality'] for r in support}),
                        'evidence_ids': [r['item_id'] for r in evidence]})
    return samples


def check_exclusive_gpu():
    import os
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    foreign = {int(p.strip()) for p in result.splitlines() if p.strip().isdigit()}-{os.getpid()}
    if foreign:
        raise RuntimeError(f'GPU contention detected; finish collection first: {sorted(foreign)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--image-soft-tokens', type=int, default=1120)
    parser.add_argument('--max-context', type=int, default=16384)
    parser.add_argument('--heads', type=Path, help='Existing head_training.json; compare every checkpoint on both modality metadata settings')
    parser.add_argument('--atol', type=float, default=0.)
    parser.add_argument('--rtol', type=float, default=0.)
    args = parser.parse_args()
    if args.limit < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error('limit/repeats must be positive and warmup nonnegative')
    check_exclusive_gpu()
    samples = diagnostic_inputs(args.data, args.limit)
    if not samples:
        raise ValueError('No diagnostic inputs')
    import torch
    import transformers
    backend = V11Backend(str(args.model), image_soft_tokens=args.image_soft_tokens, max_context=args.max_context)
    arms = {'full': backend.encode_state,
            'backbone': lambda messages: fast_encode_state(backend, messages),
            'three_rows': lambda messages: fast_encode_state(backend, messages, selected_token_logits=True)}
    for _ in range(args.warmup):
        for call in arms.values():
            call(samples[0]['messages'])
    heads = []
    if args.heads:
        from lsdrag.v11_training import load_head, infer_head
        from lsdrag.v11_protocol import state_features
        for spec in json.loads(args.heads.read_text())['runs']:
            heads.append((spec['checkpoint'], load_head(spec['checkpoint'])))
    rows = []
    started = time.time()
    for number, sample in enumerate(samples):
        check_exclusive_gpu()
        for repeat in range(args.repeats):
            # Rotate and reverse ordering across input/repeat to limit warm/order bias.
            names = list(arms)
            offset = (number+repeat)%len(names)
            names = names[offset:]+names[:offset]
            if (number+repeat)%2:
                names.reverse()
            results = {name: arms[name](sample['messages']) for name in names}
            reference = results['full']
            comparisons = {}
            for name in ('backbone', 'three_rows'):
                candidate = results[name]
                difference = np.abs(candidate['feature']-reference['feature'])
                comparison = {'feature_max_abs_delta': float(difference.max()),
                              'feature_exact': bool(np.array_equal(candidate['feature'], reference['feature'])),
                              'feature_allclose': bool(np.allclose(candidate['feature'], reference['feature'], atol=args.atol, rtol=args.rtol))}
                if name == 'three_rows':
                    expected = np.asarray(reference['probs'])
                    actual = np.asarray(candidate['probs'])
                    ordered = np.sort(expected)
                    comparison.update(probs_max_abs_delta=float(np.max(np.abs(expected-actual))),
                                      full_probs=expected.tolist(), selected_probs=actual.tolist(),
                                      full_action=int(expected.argmax()), selected_action=int(actual.argmax()),
                                      action_disagreement=bool(expected.argmax()!=actual.argmax()),
                                      full_top2_probability_margin=float(ordered[-1]-ordered[-2]))
                if heads:
                    checks = []
                    for checkpoint_path, head in heads:
                        for modality in ('text', 'image'):
                            expected_q = infer_head(head, state_features(reference['feature'], modality))
                            actual_q = infer_head(head, state_features(candidate['feature'], modality))
                            checks.append({'checkpoint': checkpoint_path, 'modality_metadata': modality,
                                           'q_max_abs_delta': float(np.max(np.abs(expected_q-actual_q))),
                                           'q_allclose': bool(np.allclose(expected_q, actual_q, atol=args.atol, rtol=args.rtol)),
                                           'action_disagreement': bool(expected_q.argmax()!=actual_q.argmax())})
                    comparison['head_checks'] = checks
                comparisons[name] = comparison
            row = {k: sample[k] for k in ('qid', 'input_hash', 'modalities', 'evidence_ids')}
            row.update(repeat=repeat, order=names, comparisons=comparisons,
                       usages={name: result['usage'] for name, result in results.items()})
            rows.append(row)
        print(json.dumps({'completed_inputs': number+1, 'total_inputs': len(samples)}), flush=True)
    feature_passed = all(c['feature_allclose'] for row in rows for c in row['comparisons'].values())
    disagreements = sum(row['comparisons']['three_rows']['action_disagreement'] for row in rows)
    head_checks = [check for row in rows for comparison in row['comparisons'].values() for check in comparison.get('head_checks', [])]
    heads_passed = all(check['q_allclose'] and not check['action_disagreement'] for check in head_checks)
    probability_passed = all(row['comparisons']['three_rows']['probs_max_abs_delta'] <= 0.005 for row in rows)
    gate = {'passed': bool(feature_passed and disagreements == 0 and heads_passed and probability_passed), 'hidden_allclose': feature_passed,
            'selected_row_probability_passed': probability_passed, 'probability_atol': 0.005,
            'selected_row_action_disagreements': disagreements, 'head_parity_passed': heads_passed,
            'head_checkpoints': len(heads), 'head_checks': len(head_checks),
            'thresholds_fixed_before_probe': True, 'atol': args.atol, 'rtol': args.rtol}
    timings = {name: {key: {'median': float(np.median([row['usages'][name][key] for row in rows])),
                           'mean': float(np.mean([row['usages'][name][key] for row in rows]))}
                      for key in ('wall_seconds', 'preprocess_seconds', 'prefill_seconds')}
               for name in arms}
    root = Path(__file__).resolve().parents[1]
    sources = ['scripts/probe_v11_fast_readout.py', 'src/lsdrag/v11_fast_readout.py',
               'src/lsdrag/v11_backend.py', 'src/lsdrag/hf_reader.py', 'src/lsdrag/v11_protocol.py']
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv'], text=True).strip()
    result = {'diagnostic_only': True, 'gold_use': 'support IDs select equivalence inputs only; no answer/policy labels in prompts',
              'passed': gate['passed'], 'gate': gate, 'feature_allclose_passed': feature_passed,
              'action_disagreements': disagreements, 'comparisons': len(rows), 'inputs': len(samples),
              'warmups_per_arm': args.warmup, 'repeats': args.repeats, 'atol': args.atol, 'rtol': args.rtol,
              'timings': timings, 'rows': rows, 'hardware': hardware,
              'torch': torch.__version__, 'transformers': transformers.__version__, 'model_path': str(args.model),
              'image_soft_tokens': args.image_soft_tokens, 'max_context': args.max_context,
              'source_hashes': {path: sha256(root/path) for path in sources},
              'data_hashes': {path: sha256(args.data/path) for path in ('manifest.json', 'train_questions.jsonl', 'gold.jsonl', 'corpus.jsonl')},
              'model_config_sha256': sha256(args.model/'config.json'), 'elapsed_seconds': time.time()-started,
              'limitation': 'Agreement on this fixed diagnostic set does not prove universal or bitwise GEMM equivalence.'}
    if args.heads:
        result['heads_manifest_sha256'] = sha256(args.heads)
        result['head_checkpoint_hashes'] = {path: sha256(path) for path, _ in heads}
    dump_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('passed', 'feature_allclose_passed', 'action_disagreements', 'comparisons', 'timings')}), flush=True)
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
