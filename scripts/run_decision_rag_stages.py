"""Decision RAG staged diagnosis and six-action BC; original argument generation.

Shared-GPU functional runs report raw costs without controlled latency claims.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import os
import time
import tarfile
import subprocess

import numpy as np

from lsdrag.agent import Agent, AgentConfig
from lsdrag.head_experiment import GeneratedChoiceReader, build_head_feature_contract, experiment_prompt, train_behavior_heads, validate_teacher_artifact, infrastructure_tool_errors, validate_completed_prefix
from lsdrag.io import dump_json, read_jsonl, sha256
from lsdrag.retrieval import Retrieval, VectorIndex
from lsdrag.tool_registry import synchronize_prompt, tool_contract_sha256, tool_schemas
from lsdrag.v11_scoring import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model-path', 'corpus', 'indexes', 'cache-dir', 'train-questions', 'dev-questions', 'gold', 'prompt', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--backup', type=Path)
    p.add_argument('--dataset-manifest', type=Path)
    p.add_argument('--phase', choices=('diagnose', 'teacher', 'train', 'evaluate'), required=True)
    p.add_argument('--scorer', type=Path, default=Path('/content/local_fast/lsdrag-v11/scorer'))
    p.add_argument('--embedding-python', type=Path, default=Path('/content/local_fast/lsdrag-v11/embedding-env/bin/python'))
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--min-free-gb', type=float, default=32.)
    p.add_argument('--max-loops', type=int, default=8)
    p.add_argument('--max-output-tokens', type=int, default=256)
    a = p.parse_args()
    if a.limit < 1 or a.start < 0 or a.start+a.limit > 100:
        p.error('Fixed teacher blocks must stay within the first 100 questions')
    if (a.output/'DISCARDED').exists() or any(a.output.glob('DISCARDED*')) or (a.output/'INVALID_PROTOCOL_ATTEMPT.json').exists():
        raise ValueError('Discarded experiment directory cannot be reused')
    a.output.mkdir(parents=True, exist_ok=True)
    train, dev = read_jsonl(a.train_questions), read_jsonl(a.dev_questions)
    if any(set(q) != {'qid', 'question'} for q in train+dev):
        raise ValueError('Question files must contain only qid and question')
    train_ids, dev_ids = {q['qid'] for q in train}, {q['qid'] for q in dev}
    if train_ids & dev_ids or len(train_ids) != len(train) or len(dev_ids) != len(dev):
        raise ValueError('Overlapping or duplicate train/development questions')
    gold = {r['qid']: r for r in read_jsonl(a.gold)}
    train_groups = {gold[q]['source_group'] for q in train_ids if 'source_group' in gold[q]}
    dev_groups = {gold[q]['source_group'] for q in dev_ids if 'source_group' in gold[q]}
    if train_groups & dev_groups:
        raise ValueError('Source groups cross train/development boundary')
    config = AgentConfig(max_loops=a.max_loops, max_output_tokens=a.max_output_tokens,
                         context_limit=32768, max_images=8, image_soft_tokens=1120,
                         image_delivery_policy='search-linked', tool_profile='multimodal_full')
    tools = tool_schemas('G1', config.image_delivery_policy, config.tool_profile)
    prompt = experiment_prompt(a.prompt.read_text())
    effective = synchronize_prompt(prompt, tools, config.image_delivery_policy)
    contract = build_head_feature_contract(a.model_path, config, tools, [hashlib.sha256(effective.encode()).hexdigest()])
    root = Path(__file__).resolve().parents[1]
    source_names = ['scripts/run_decision_rag_stages.py']+['src/lsdrag/'+s+'.py' for s in
        ('head_experiment', 'direct_reader', 'direct_contract', 'agent', 'tool_registry', 'hf_reader', 'retrieval', 'context', 'gemma4_arguments')]
    manifest = {'experiment': 'decision-rag-v13-original-arguments', 'controller_baseline': 'generated letter plus common argument generation; not native A-RAG serialization',
        'feature_contract': contract, 'config': asdict(config), 'tool_contract_sha256': tool_contract_sha256(tools),
        'data_hashes': {str(path): sha256(path) for path in (a.train_questions, a.dev_questions, a.gold, a.prompt,
                              a.corpus/'chunks.jsonl', a.corpus/'figures.jsonl',
                              a.indexes/'gemma_text'/'manifest.json', a.indexes/'gemma_image'/'manifest.json')},
        'source_hashes': {name: sha256(root/name) for name in source_names},
        'shared_gpu': True, 'controlled_latency_claims': False, 'train_question_order': [q['qid'] for q in train[:100]],
        'development_question_order': [q['qid'] for q in dev[:20]], 'coverage_minimum_distinct_questions': 5,
        'seeds': [20261015, 20261016, 20261017], 'epochs': 100, 'objective': 'plain unweighted behavior cloning CE'}
    if a.dataset_manifest:
        manifest['data_hashes'][str(a.dataset_manifest)] = sha256(a.dataset_manifest)
    else:
        candidate = a.train_questions.parent/'manifest.json'
        if candidate.exists():
            manifest['data_hashes'][str(candidate)] = sha256(candidate)
    def save(relative, value):
        path = a.output/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+'.tmp')
        dump_json(temporary, value)
        os.replace(temporary, path)

    def mirror(relative):
        pass  # Bundle at phase boundaries; diagnostic bundles every question.
    manifest_path = a.output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Experiment source/data/feature contract changed; use a new output directory')
    save('manifest.json', manifest)
    import torch
    if a.phase == 'train':
        records, features, teacher_results = [], [], []
        for q in train[:100]:
            path = a.output/'teacher'/f'{q["qid"]}.pt'
            if path.exists():
                artifact = torch.load(path, map_location='cpu', weights_only=True)
                validate_teacher_artifact(artifact, q['qid'], contract)
                result_path = path.with_suffix('.json')
                if not result_path.exists():
                    raise ValueError('Teacher trajectory is incomplete')
                teacher_results.append(json.loads(result_path.read_text()))
                records.extend(artifact['records'])
                features.extend(artifact['features'].numpy())
        if len(teacher_results) < 60:
            save('training_gate.json', {'passed': False, 'completed_teacher_questions': len(teacher_results), 'minimum_completed_teacher_questions': 60})
            raise ValueError('At least 60 completed teacher questions required; 20-question block is diagnostic only')
        validate_completed_prefix([r['qid'] for r in teacher_results], [q['qid'] for q in train])
        result = train_behavior_heads(records, features, a.output/'heads', contract, tools)
        result['teacher_statistics'] = {'completed_questions': len(teacher_results),
            'mean_em': float(np.mean([r['em'] for r in teacher_results])),
            'mean_f1': float(np.mean([r['f1'] for r in teacher_results])),
            'questions_with_errors': sum(bool(r['errors']) for r in teacher_results),
            'error_count': sum(len(r['errors']) for r in teacher_results),
            'status_histogram': {status: sum(r['status'] == status for r in teacher_results) for status in {r['status'] for r in teacher_results}}}
        save('training.json', result)
        if a.backup:
            for path in (a.output/'heads').glob('*'):
                mirror(path.relative_to(a.output))
        print(json.dumps({'phase': 'train', 'gate': result['gate'], 'runs': len(result['runs'])}), flush=True)
        return
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
    if float(free.splitlines()[0])/1024 < a.min_free_gb:
        raise RuntimeError('Not enough free GPU memory for functional run')
    reader = GeneratedChoiceReader(str(a.model_path), config, tools, contract)
    from lsdrag.v11_embedding_worker import V11ProcessEmbedder
    worker = V11ProcessEmbedder(a.embedding_python, a.cache_dir)
    try:
        indexes = {name: VectorIndex(a.indexes/name, a.corpus/('figures.jsonl' if name.endswith('image') else 'chunks.jsonl'))
                   for name in ('gemma_text', 'gemma_image')}
        retrieval = Retrieval(a.corpus, indexes, {'gemma': worker}, 'G1')
        def run(question, controller):
            controller.current_qid, controller.natural_loop = question['qid'], 0
            retrieval.query_cache.clear()
            result = Agent(retrieval, getattr(controller, 'agent_prompt', prompt), config, controller).run(question['qid'], question['question'])
            unexpected = [error for error in result['errors'] if not error['error'].startswith('DirectOutputError:')]
            unknown_cost = any(call['usage'].get('generation_cost_unknown') for call in result['reader_calls'])
            tool_failures = infrastructure_tool_errors(result['trajectory'])
            if unexpected or unknown_cost or tool_failures:
                save(Path('infrastructure_failures')/f'{question["qid"]}.json', result)
                raise RuntimeError('Infrastructure/context/model failure; preserved trajectory and halted without complete teacher artifact')
            result.update(score(result['answers'], gold[question['qid']]['answers'], source_dir=a.scorer))
            return result
        if a.phase == 'diagnose':
            from lsdrag.hf_reader import HFReader
            native = HFReader.__new__(HFReader)
            native.__dict__ = reader.__dict__.copy()
            native.head = None
            native.agent_prompt = a.prompt.read_text()
            policies = [('native', native), ('generated', reader),
                        ('first_token', GeneratedChoiceReader.from_reader(reader, 'first_token'))]
            for number, q in enumerate(train[a.start:a.start+a.limit], a.start):
                schedule = list(policies)
                random.Random(20261015+number).shuffle(schedule)
                for name, controller in schedule:
                    relative = Path('diagnostic')/name/f'{q["qid"]}.json'
                    if (a.output/relative).exists():
                        prior = json.loads((a.output/relative).read_text())
                        if prior['qid'] != q['qid'] or prior['policy'] != name:
                            raise ValueError('Diagnostic resume identity mismatch')
                        continue
                    states, vectors = [], []
                    controller.state_sink = (lambda record, feature: (states.append(record), vectors.append(feature))) if name == 'generated' else None
                    result = run(q, controller)
                    result['policy'] = name
                    # Persist fresh generated trajectories as same-contract teacher data.
                    if name == 'generated':
                        for state in states: state['episode_status'] = result['status']
                        width = reader.model.config.get_text_config().hidden_size
                        features = np.asarray(vectors, dtype=np.float32).reshape(-1, width)
                        teacher = a.output/'teacher'/f'{q["qid"]}.pt'
                        teacher.parent.mkdir(parents=True, exist_ok=True)
                        temporary = teacher.with_suffix('.pt.tmp')
                        torch.save({'records': states, 'features': torch.from_numpy(features),
                                    'feature_contract': contract, 'trajectory_status': result['status']}, temporary)
                        save(Path('teacher')/f'{q["qid"]}.json', result)
                        os.replace(temporary, teacher)
                    save(relative, result)
                    save('progress.json', {'phase':'diagnose', 'question_index':number,
                         'last_policy':name, 'last_qid':q['qid'], 'updated_epoch':time.time()})
                    print(json.dumps({'phase':'diagnose','index':number,'policy':name,
                          'qid':q['qid'],'f1':result['f1'],'status':result['status']}), flush=True)
                # Bundle every completed question, no per-file Drive writes.
                if a.backup:
                    bundle = a.output.parent/(a.output.name+'_latest.tar.gz')
                    temporary = bundle.with_suffix('.tmp')
                    with tarfile.open(temporary, 'w:gz') as archive:
                        archive.add(a.output, arcname=a.output.name)
                    os.replace(temporary, bundle)
                    a.backup.mkdir(parents=True, exist_ok=True)
                    backup_temp = a.backup/(bundle.name+'.tmp')
                    shutil.copy2(bundle, backup_temp)
                    os.replace(backup_temp, a.backup/bundle.name)
            return
        if a.phase == 'teacher':
            for n, q in enumerate(train[a.start:a.start+a.limit], 1):
                relative = Path('teacher')/f'{q["qid"]}.pt'
                if (a.output/relative).exists() and (a.output/relative.with_suffix('.json')).exists():
                    validate_teacher_artifact(torch.load(a.output/relative, map_location='cpu', weights_only=True), q['qid'], contract)
                    continue
                states, vectors = [], []
                reader.state_sink = lambda record, feature: (states.append(record), vectors.append(feature))
                result = run(q, reader)
                for state in states:
                    state['episode_status'] = result['status']
                width = reader.model.config.get_text_config().hidden_size
                features = np.asarray(vectors, dtype=np.float32).reshape(-1, width)
                path = a.output/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                save(Path('teacher')/f'{q["qid"]}.json', result)
                torch.save({'records': states, 'features': torch.from_numpy(features), 'feature_contract': contract,
                            'trajectory_status': result['status']}, path)
                mirror(relative)
                print(json.dumps({'phase': 'teacher', 'done': n, 'qid': q['qid'], 'states': len(states), 'status': result['status']}), flush=True)
        else:
            trained = json.loads((a.output/'training.json').read_text())
            if not trained['gate']['passed']:
                raise ValueError('Six-action coverage gate failed; evaluation not authorized by protocol')
            policies = [('generated', reader), ('first_token', GeneratedChoiceReader.from_reader(reader, 'first_token'))]
            policies += [(f'head_s{spec["seed"]}', GeneratedChoiceReader.from_reader(reader, 'head', spec['checkpoint'])) for spec in trained['runs']]
            results = []
            for number, q in enumerate(dev[:20]):
                schedule = list(policies)
                random.Random(20261015+number).shuffle(schedule)
                for name, controller in schedule:
                    heldout_states, heldout_vectors = [], []
                    controller.state_sink = (lambda record, feature: (heldout_states.append(record), heldout_vectors.append(feature))) if name == 'generated' else None
                    relative = Path('development')/name/f'{q["qid"]}.json'
                    path = a.output/relative
                    if path.exists():
                        result = json.loads(path.read_text())
                    else:
                        result = run(q, controller)
                        result['policy'] = name
                        if name == 'generated':
                            feature_relative = Path('development_teacher_states')/f'{q["qid"]}.pt'
                            feature_path = a.output/feature_relative
                            feature_path.parent.mkdir(parents=True, exist_ok=True)
                            width = reader.model.config.get_text_config().hidden_size
                            torch.save({'records': heldout_states, 'features': torch.from_numpy(np.asarray(heldout_vectors, dtype=np.float32).reshape(-1, width)), 'feature_contract': contract}, feature_path)
                            mirror(feature_relative)
                        save(relative, result)
                    results.append(result)
                print(json.dumps({'phase': 'evaluate', 'done': number+1}), flush=True)
            summary = {name: {'n': sum(r['policy'] == name for r in results),
                       'em': float(np.mean([r['em'] for r in results if r['policy'] == name])),
                       'f1': float(np.mean([r['f1'] for r in results if r['policy'] == name])),
                       'valid_answer_fraction': float(np.mean([r['answer_format_valid'] for r in results if r['policy'] == name]))}
                       for name, _ in policies}
            agreement = {}
            heldout = [torch.load(path, map_location='cpu', weights_only=True) for path in sorted((a.output/'development_teacher_states').glob('*.pt'))]
            for spec in trained['runs']:
                saved = torch.load(spec['checkpoint'], map_location='cpu', weights_only=True)
                correct, total = 0, 0
                for artifact in heldout:
                    logits = artifact['features'] @ saved['weight'].T + saved['bias']
                    labels = torch.tensor([record['index'] for record in artifact['records']])
                    correct += int((logits.argmax(-1) == labels).sum()); total += len(labels)
                agreement[str(spec['seed'])] = {'correct': correct, 'states': total, 'accuracy': correct/total if total else None, 'scope': 'same generated-teacher heldout states; forced FINAL excluded'}
            save('evaluation.json', {'policies': summary, 'heldout_teacher_agreement': agreement, 'scope': 'paired development behavior-cloning pilot; no test/novelty/latency claim'})
    finally:
        worker.close()


if __name__ == '__main__':
    main()
