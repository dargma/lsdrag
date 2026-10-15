"""Development-only acceleration amendment; retains every frozen original arm.

No collection/training manifests or checkpoints are changed. The equivalence
probe and amendment must be fixed before the first development outcome exists.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

import numpy as np

from lsdrag.io import dump_json, read_jsonl, sha256
from lsdrag.v11_fast_readout import fast_encode_state
from lsdrag.v11_training import load_head, infer_head

REPO = Path(__file__).resolve().parents[1]
BASE_PATH = REPO / 'scripts/run_v11_pilot.py'
spec = importlib.util.spec_from_file_location('_v11_original_pilot', BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def frozen_amendment(args, runs):
    plan_path = REPO / 'lab/registry/jev_v1.1/execution_plan.json'
    plan = json.loads(plan_path.read_text())
    if sha256(args.data/'manifest.json') != plan['data_manifest_sha256']:
        raise ValueError('Prepared data changed after plan freeze')
    data = json.loads((args.data/'manifest.json').read_text())
    for filename, digest in data['files'].items():
        if sha256(args.data/filename) != digest:
            raise ValueError('Prepared data changed: '+filename)
    identity = {'plan_sha256': sha256(plan_path),
                'data_manifest_sha256': sha256(args.data/'manifest.json')}
    original = json.loads((args.output/'run_manifest.json').read_text())
    if any(original.get(k) != v for k,v in identity.items()):
        raise ValueError('Existing run has another plan/data identity')
    probe = json.loads(args.probe.read_text())
    if probe.get('gate', {}).get('passed') is not True:
        raise ValueError('A passing acceleration equivalence probe is required')
    if probe['gate'].get('head_checkpoints') != 12 or probe.get('heads_manifest_sha256') != sha256(args.output/'head_training.json'):
        raise ValueError('Probe must verify the same twelve learned checkpoints')
    if probe.get('data_hashes', {}).get('manifest.json') != sha256(args.data/'manifest.json'):
        raise ValueError('Probe used another dataset')
    if probe.get('model_config_sha256') != sha256(args.model/'config.json'):
        raise ValueError('Probe used another reader configuration')
    if (probe.get('atol'), probe.get('rtol'), probe.get('image_soft_tokens'), probe.get('max_context')) != (0, 0, 1120, 16384):
        raise ValueError('Probe differs from the fixed exact-hidden and multimodal configuration')
    if any(probe.get('head_checkpoint_hashes', {}).get(r['checkpoint']) != sha256(Path(r['checkpoint'])) for r in runs):
        raise ValueError('Checkpoint weights changed after equivalence probe')
    sources = ['scripts/run_v11_pilot.py', 'scripts/run_v11_accelerated_evaluation.py'] + [
        'src/lsdrag/'+name for name in ['v11_fast_readout.py', 'v11_backend.py',
        'v11_protocol.py', 'v11_scoring.py', 'v11_training.py',
        'v11_embedding_worker.py', 'embeddings.py', 'hf_reader.py']]
    source_hashes = {f: sha256(REPO/f) for f in sources}
    for required in ['src/lsdrag/v11_fast_readout.py', 'src/lsdrag/v11_backend.py']:
        if probe.get('source_hashes', {}).get(required) != source_hashes[required]:
            raise ValueError('Probe does not verify current source: '+required)
    # A probe can certify more dependencies; require every certified file unchanged.
    for filename, digest in probe['source_hashes'].items():
        if not (REPO/filename).is_file() or sha256(REPO/filename) != digest:
            raise ValueError('Source changed after equivalence probe: '+filename)
    if len(runs) != 12:
        raise ValueError('All twelve predetermined checkpoints are required')
    signatures = {(r['lambda'], r['beta'], r['seed']) for r in runs}
    expected = {(lam, beta, seed) for lam in plan['training']['lambdas']
                for beta in plan['training']['betas'] for seed in plan['training']['seeds']}
    if len(signatures) != 12 or signatures != expected:
        raise ValueError('Checkpoint grid differs from frozen plan')
    policies = ['first_token', 'generated', 'always_stop', 'always_text', 'always_image']
    policies += [f'head_l{r["lambda"]}_b{r["beta"]}_s{r["seed"]}' for r in runs]
    policies += ['fast_first_token']
    policies += [f'fast_head_l{r["lambda"]}_b{r["beta"]}_s{r["seed"]}' for r in runs]
    manifest = {**identity, 'phase': 'evaluate_acceleration_amendment',
        'probe_path': str(args.probe.resolve()), 'probe_sha256': sha256(args.probe),
        'source_hashes': source_hashes, 'policies': policies,
        'checkpoints': {str(Path(r['checkpoint']).resolve()): sha256(Path(r['checkpoint'])) for r in runs},
        'head_training_sha256': sha256(args.output/'head_training.json'),
        'preregistered_amendment_sha256': sha256(REPO/'lab/registry/jev_v1.1/head_acceleration_amendment.json'),
        'model': str(args.model.resolve()), 'scorer': str(args.scorer.resolve()),
        'timing_rule': 'Decision wall time includes preprocessing, backbone, feature transfer, state metadata, CPU head inference, argmax. Fresh query/answer execution for every arm.',
        'interpretation': 'Implementation acceleration amendment, not a new learned method; no development seed selection.'}
    path = args.output/'acceleration_amendment_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Acceleration amendment identity changed; preserve this run')
    else:
        if any((args.output/'development').glob('*/*.json')) or (args.output/'development_summary.json').exists():
            raise ValueError('Cannot add acceleration arms after development outcomes exist')
        # Exclusive create records amendment before model loading or development execution.
        with path.open('x') as stream:
            json.dump(manifest, stream, indent=2)
        if args.backup:
            dump_json(args.backup/path.name, manifest)
    return manifest


class AcceleratedExperiment(base.Experiment):
    def evaluate(self, runs):
        heads = [(r, load_head(r['checkpoint'])) for r in runs]
        policies = [(name, None) for name in ['first_token', 'generated', 'always_stop',
                                            'always_text', 'always_image']]
        policies += [(f'head_l{r["lambda"]}_b{r["beta"]}_s{r["seed"]}', head) for r,head in heads]
        policies += [('fast_first_token', None)]
        policies += [(f'fast_head_l{r["lambda"]}_b{r["beta"]}_s{r["seed"]}', head) for r,head in heads]
        predictions = []
        for number, question in enumerate(self.dev, 1):
            self.check_gpu()
            modality, initial = self.initialize(question)
            evidence = initial['evidence']
            schedule = list(policies)
            random.Random(20261015+number).shuffle(schedule)
            for name, head in schedule:
                relative = f'development/{name}/{question["qid"]}.json'
                path = self.args.output/relative
                if path.exists():
                    predictions.append(json.loads(path.read_text())); continue
                start = time.perf_counter(); decision_seconds = 0.
                if head is not None:
                    tick = time.perf_counter()
                    messages = self.messages(question['question'], evidence, base.DECISION)
                    encoded = (fast_encode_state(self.reader, messages) if name.startswith('fast_')
                               else self.reader.encode_state(messages))
                    values = infer_head(head, base.state_features(encoded['feature'], modality))
                    index = int(np.argmax(values))
                    decision_seconds = time.perf_counter()-tick
                    control = {'qvalues': values.tolist(), 'state_usage': encoded['usage'],
                               'readout': 'fast' if name.startswith('fast_') else 'original'}
                elif name == 'fast_first_token':
                    tick = time.perf_counter()
                    encoded = fast_encode_state(self.reader,
                        self.messages(question['question'], evidence, base.DECISION),
                        selected_token_logits=True)
                    index = int(np.argmax(encoded['probs']))
                    decision_seconds = time.perf_counter()-tick
                    control = {'index': index, 'action_probs': encoded['probs'], 'usage': encoded['usage']}
                elif name in ('first_token', 'generated'):
                    tick = time.perf_counter()
                    control = self.reader.decide(self.messages(question['question'], evidence, base.DECISION),
                                                 mode=name, max_new_tokens=32)
                    index = control['index']; decision_seconds = time.perf_counter()-tick
                    control = {k:v for k,v in control.items() if k != 'feature'}
                else:
                    index = {'always_stop': 0, 'always_text': 1, 'always_image': 2}[name]; control = {}
                query = None
                if index:
                    self.retriever.query_cache.clear()
                    query = self.acquire(question['question'], evidence, 'text' if index==1 else 'image',
                                         exclude=initial['result_ids'])
                messages = self.messages(question['question'], evidence+(query['evidence'] if query else []), base.ANSWER)
                result = self.generate(messages, 128, '{"answers":')
                parsed = base.parsed_answers(result['text'])
                continuation_seconds = time.perf_counter()-start
                row = {'qid': question['qid'], 'source_group': self.gold[question['qid']]['source_group'],
                    'policy': name, 'action': base.ACTIONS[index], 'answers': parsed['answers'],
                    'answer_format_valid': parsed['answer_format_valid'], 'raw': result['text'],
                    'initial_ids': initial['result_ids'], 'query': query, 'control': control,
                    'initial_seconds': initial['cost_seconds'], 'decision_seconds': decision_seconds,
                    'continuation_seconds': continuation_seconds,
                    'total_seconds': initial['cost_seconds']+continuation_seconds,
                    **self.score(parsed['answers'], question['qid'])}
                self.save(relative, row); predictions.append(row)
            print(json.dumps({'phase':'accelerated_development', 'done':number, 'n':len(self.dev),
                              'arms':len(policies)}), flush=True)
            self.check_gpu()
        summary = {}
        for name,_ in policies:
            rows = [r for r in predictions if r['policy']==name]
            summary[name] = {'n':len(rows), **{k:statistics.mean(r[k] for r in rows)
                for k in ['f1','em','total_seconds','decision_seconds']},
                'actions':{a:sum(r['action']==a for r in rows) for a in base.ACTIONS}}
        self.save('development_summary.json', summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['data','scorer','model','hf','output','backup']:
        parser.add_argument('--'+name, type=Path, required=name!='backup')
    parser.add_argument('--phase', choices=['evaluate'], default='evaluate')
    parser.add_argument('--probe', type=Path, required=True)
    args = parser.parse_args()
    if (args.output/"DISCARDED.json").exists():
        raise ValueError("This run was discarded by the user; do not resume or reuse its labels")
    trained = json.loads((args.output/'head_training.json').read_text())
    amendment = frozen_amendment(args, trained['runs'])
    if not json.loads((args.output/'oracle_summary.json').read_text())['gate']['passed']:
        raise ValueError('Reader gate did not pass')
    questions = read_jsonl(args.data/'train_questions.jsonl')
    rows = [json.loads((args.output/'train'/f'{q["qid"]}.json').read_text()) for q in questions]
    gate = base.contrast_summary(rows)
    if gate['contrast_questions']<30 or gate['contrast_source_groups']<10:
        raise ValueError('Access gate failed before evaluation')
    experiment = AcceleratedExperiment(args)
    experiment.warmup(); experiment.setup_retriever()
    experiment.evaluate(trained['runs'])
    experiment.save('status.json', {'stage1':'completed','stage2':'trained_and_development_evaluated',
        'acceleration_amendment_sha256':sha256(args.output/'acceleration_amendment_manifest.json'),
        'evaluated_arms':len(amendment['policies']),
        'note':'controlled continuation development pilot; no final-test or new-method claim'})

if __name__ == '__main__':
    main()
