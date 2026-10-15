"""Preregistered stage 1 diagnostics, then gated v1.1 head training/evaluation."""
import argparse
import hashlib
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from lsdrag.io import dump_json, read_jsonl, sha256
from lsdrag.v11_backend import ACTIONS, V11Backend, V11Retriever
from lsdrag.v11_protocol import (ANSWER, DECISION, contrast_summary, effects, input_hash,
                                observation_messages, parsed_answers, state_features)


class Experiment:
    def __init__(self, args):
        self.args = args
        self.items = {r['item_id']: r for r in read_jsonl(args.data/'corpus.jsonl')}
        self.gold = {r['qid']: r for r in read_jsonl(args.data/'gold.jsonl')}
        self.train = read_jsonl(args.data/'train_questions.jsonl')
        self.dev = read_jsonl(args.data/'development_questions.jsonl')
        self.stage1 = read_jsonl(args.data/'stage1_questions.jsonl')
        if any(set(q) != {'qid', 'question'} for q in self.train+self.dev):
            raise ValueError('Private fields in question input')
        from lsdrag.v11_scoring import score
        self.scorer = lambda p, g: score(p, g, source_dir=args.scorer)
        self.reader = V11Backend(str(args.model), image_soft_tokens=1120, max_context=16384)
        self.retriever = None
        self.answer_cache = {}
        self.query_cache = {}
        self.counter = 0

    def check_gpu(self):
        import os
        raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'], text=True)
        foreign = {int(p.strip()) for p in raw.splitlines() if p.strip().isdigit()}-{os.getpid()}
        if foreign:
            self.save('timing_invalid.json', {'foreign_gpu_pids': sorted(foreign),
                                              'reason': 'GPU contention; timing run stopped'})
            raise RuntimeError('GPU contention detected; preserve results and stop')

    def save(self, relative, value):
        dump_json(self.args.output/relative, value)
        if self.args.backup:
            dump_json(self.args.backup/relative, value)

    def score(self, answers, qid):
        return self.scorer(answers, self.gold[qid]['answers'])

    def generate(self, messages, budget, prefix=''):
        result = self.reader.generate(messages, max_new_tokens=budget, prefix=prefix)
        self.counter += 1
        return result

    def answer(self, question, evidence):
        messages = observation_messages(question, evidence, ANSWER)
        digest = input_hash(messages)
        if digest in self.answer_cache:
            return {**self.answer_cache[digest], 'reused_identical_input': True}
        result = self.generate(messages, 128, '{"answers":')
        record = {**parsed_answers(result['text']), 'raw': result['text'],
                  'usage': result['usage'], 'input_hash': digest,
                  'evidence_ids': [e['item_id'] for e in evidence],
                  'cost_seconds': result['usage']['wall_seconds']}
        self.answer_cache[digest] = record
        return record

    def acquire(self, question, evidence, modality, exclude=()):
        instruction = ('Write ONE concise search query to retrieve a '+modality+
                       ' evidence item needed to answer the original question. '
                       'Use the evidence to identify specific entities or missing facts. '
                       'Return only the search query, without quotation marks or explanation.')
        messages = observation_messages(question, evidence, instruction)
        start = time.perf_counter()
        result = self.generate(messages, 48)
        query = result['text'].strip().strip('"')
        if not query:
            query = question  # Mechanical empty-output fallback, recorded below.
        retrieved = self.retriever.search(query, modality, k=1, exclude=exclude)
        return {'query': query, 'query_input_hash': input_hash(messages),
                'query_usage': result['usage'], 'evidence': retrieved,
                'result_ids': [r['item_id'] for r in retrieved],
                'cost_seconds': time.perf_counter()-start,
                'empty_query_fallback': not bool(result['text'].strip())}

    def initialize(self, question):
        # Outcome-blind fixed behavior, not support modality or a learned gold router.
        modality = 'image' if int(hashlib.sha256(question['qid'].encode()).hexdigest(), 16)%2 else 'text'
        acquisition = self.acquire(question['question'], [], modality)
        if len(acquisition['evidence']) != 1:
            raise ValueError('Acquisition must return exactly one unique item')
        return modality, acquisition

    def warmup(self):
        result = self.generate([{'role': 'user', 'content': 'Reply OK.'}], 8)
        decision = self.reader.decide(observation_messages('What is 2 plus 2?', [], DECISION))
        tokenizer = self.reader.processor.tokenizer
        template = self.reader.processor.apply_chat_template(
            [{'role': 'user', 'content': 'Choose A, B or C.'}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        before = tokenizer.encode(template, add_special_tokens=False)
        suffixes = {a: tokenizer.encode(template+a, add_special_tokens=False)[len(before):]
                    for a in 'ABC'}
        if any(v != [self.reader.label_ids[i]] for i, v in enumerate(suffixes.values())):
            raise ValueError('Contextual decision labels are not the intended single tokens')
        self.save('warmup.json', {'text': result['text'], 'decision': decision['action'],
                  'label_ids': self.reader.label_ids, 'contextual_label_ids': suffixes})

    def diagnose(self):
        rows = []
        for q in self.stage1:
            self.check_gpu()
            path = self.args.output/'oracle'/f'{q["qid"]}.json'
            if path.exists():
                rows.append(json.loads(path.read_text())); continue
            evidence = [self.items[k] for k in self.gold[q['qid']]['support_ids']]
            modes = [('question_only', []), ('gold_evidence', evidence)]
            record = {'qid': q['qid'], 'source_group': self.gold[q['qid']]['source_group']}
            for name, observed in modes:
                prediction = self.answer(q['question'], observed)
                record[name] = {**prediction, **self.score(prediction['answers'], q['qid'])}
            self.save(f'oracle/{q["qid"]}.json', record); rows.append(record)
            self.check_gpu()
            print(json.dumps({'phase': 'oracle', 'done': len(rows), 'n': len(self.stage1),
                             'gold_em': record['gold_evidence']['em'],
                             'gold_f1': record['gold_evidence']['f1']}), flush=True)
        summary = {name: {key: statistics.mean(r[name][key] for r in rows)
                   for key in ('em', 'f1', 'cost_seconds')} for name in ('question_only', 'gold_evidence')}
        correct = sum(r['gold_evidence']['em'] == 1 for r in rows)
        delta = summary['gold_evidence']['f1']-summary['question_only']['f1']
        gate = {'passed': correct >= 10 and delta >= .05, 'gold_correct_questions': correct,
                'gold_f1_gain': delta, 'rule': 'gold EM>=10/40 and mean F1 gain>=.05'}
        self.save('oracle_summary.json', {'n': len(rows), 'arms': summary, 'gate': gate})
        return gate

    def setup_retriever(self):
        if self.retriever is None:
            self.retriever = V11Retriever(list(self.items.values()), self.args.output/'indexes', self.args.hf)
            self.retriever.search('initialization', 'text', k=1)

    def collect(self, questions):
        rows = []
        for q in questions:
            self.check_gpu()
            path = self.args.output/'train'/f'{q["qid"]}.json'
            if path.exists():
                rows.append(json.loads(path.read_text())); continue
            start = time.perf_counter()
            modality, initial = self.initialize(q)
            e = initial['evidence']
            record = {'qid': q['qid'], 'source_group': self.gold[q['qid']]['source_group'],
                      'initial_modality': modality, 'initial': initial,
                      'stops': [], 'features': [], 'actions': {}}
            for i in (0, 1):
                current = e if i else []
                stop = self.answer(q['question'], current)
                record['stops'].append({**stop, **self.score(stop['answers'], q['qid'])})
                encoded = self.reader.encode_state(observation_messages(q['question'], current, DECISION))
                record['features'].append(state_features(encoded['feature'], modality))
                record.setdefault('state_usage', []).append(encoded['usage'])
            # Alternate execution order deterministically; no gold-dependent schedule.
            order = ['text', 'image'] if modality == 'text' else ['image', 'text']
            for next_modality in order:
                acquisitions = [self.acquire(q['question'], e if j else [], next_modality,
                                exclude=initial['result_ids']) for j in (0, 1)]
                action = {'queries': acquisitions, 'cells': {}}
                for i, j in ((0, 0), (1, 1), (0, 1), (1, 0)):
                    evidence = (e if i else []) + acquisitions[j]['evidence']
                    prediction = self.answer(q['question'], evidence)
                    action['cells'][f'{i}{j}'] = {
                        **prediction, **self.score(prediction['answers'], q['qid']),
                        'cost_seconds': prediction['cost_seconds']+acquisitions[j]['cost_seconds']}
                action.update(effects(action))
                record['actions']['TEXT_ACQUIRE' if next_modality == 'text' else 'IMAGE_ACQUIRE'] = action
            record['collection_seconds'] = time.perf_counter()-start
            self.save(f'train/{q["qid"]}.json', record); rows.append(record)
            self.check_gpu()
            print(json.dumps({'phase': 'crossed', 'done': len(rows), 'n': len(questions),
                  'seconds': record['collection_seconds'],
                  'signal': any(abs(a[k])>=.05 for a in record['actions'].values() for k in ('N0','J'))}), flush=True)
        return rows

    def evaluate(self, runs):
        from lsdrag.v11_training import load_head, infer_head
        heads = [(r, load_head(r['checkpoint'])) for r in runs]
        predictions = []
        # All predetermined seeds/operating points; never select the winning dev seed.
        policies = [('first_token', None), ('generated', None), ('always_stop', None),
                    ('always_text', None), ('always_image', None)]
        policies += [(f'head_l{r["lambda"]}_b{r["beta"]}_s{r["seed"]}', h) for r, h in heads]
        for number, q in enumerate(self.dev, 1):
            self.check_gpu()
            modality, initial = self.initialize(q)
            e = initial['evidence']
            # Actual training-input source exposure is audited separately below.
            schedule = list(policies)
            random.Random(20261015+number).shuffle(schedule)
            for name, head in schedule:
                path = self.args.output/'development'/name/f'{q["qid"]}.json'
                if path.exists():
                    predictions.append(json.loads(path.read_text())); continue
                start = time.perf_counter()
                feature_cost = 0.
                if head is not None:
                    encoded = self.reader.encode_state(observation_messages(q['question'], e, DECISION))
                    feature_cost = encoded['usage']['wall_seconds']
                    qvalues = infer_head(head, state_features(encoded['feature'], modality))
                    index = int(np.argmax(qvalues))
                    control = {'qvalues': qvalues.tolist(), 'state_usage': encoded['usage']}
                elif name in ('first_token', 'generated'):
                    control = self.reader.decide(observation_messages(q['question'], e, DECISION), mode=name,
                                                 max_new_tokens=32)
                    index = control['index']; feature_cost = control['usage']['wall_seconds']
                    control = {k: v for k, v in control.items() if k != 'feature'}
                else:
                    index = {'always_stop': 0, 'always_text': 1, 'always_image': 2}[name]
                    control = {}
                query = None
                if index:
                    self.retriever.query_cache.clear()
                    query = self.acquire(q['question'], e, 'text' if index == 1 else 'image',
                                         exclude=initial['result_ids'])
                # Evaluation never reuses counterfactual answer cache: fresh policy execution.
                messages = observation_messages(q['question'], e+(query['evidence'] if query else []), ANSWER)
                result = self.generate(messages, 128, '{"answers":')
                parsed = parsed_answers(result['text'])
                row = {'qid': q['qid'], 'source_group': self.gold[q['qid']]['source_group'],
                       'policy': name, 'action': ACTIONS[index], 'answers': parsed['answers'],
                       'answer_format_valid': parsed['answer_format_valid'], 'raw': result['text'],
                       'initial_ids': initial['result_ids'], 'query': query, 'control': control,
                       'initial_seconds': initial['cost_seconds'], 'decision_seconds': feature_cost,
                       'continuation_seconds': time.perf_counter()-start,
                       **self.score(parsed['answers'], q['qid'])}
                row['total_seconds'] = row['initial_seconds']+row['continuation_seconds']
                self.save(f'development/{name}/{q["qid"]}.json', row); predictions.append(row)
            print(json.dumps({'phase': 'development', 'done': number, 'n': len(self.dev)}), flush=True)
            self.check_gpu()
        summary = {}
        for name, _ in policies:
            rows = [r for r in predictions if r['policy'] == name]
            summary[name] = {'n': len(rows), **{k: statistics.mean(r[k] for r in rows)
                  for k in ('f1', 'em', 'total_seconds', 'decision_seconds')},
                  'actions': {a: sum(r['action']==a for r in rows) for a in ACTIONS}}
        self.save('development_summary.json', summary)
        return summary


def main():
    p = argparse.ArgumentParser()
    for name in ('data', 'scorer', 'model', 'hf', 'output', 'backup'):
        p.add_argument('--'+name, type=Path, required=name != 'backup')
    p.add_argument('--phase', choices=['all', 'diagnose', 'collect', 'train', 'evaluate'], default='all')
    a = p.parse_args()
    plan_path = Path(__file__).resolve().parents[1]/'lab/registry/jev_v1.1/execution_plan.json'
    plan = json.loads(plan_path.read_text())
    if sha256(a.data/'manifest.json') != plan['data_manifest_sha256']:
        raise ValueError('Prepared data changed after plan freeze')
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = {'plan_sha256': sha256(plan_path), 'data_manifest_sha256': sha256(a.data/'manifest.json'),
                'code_revision': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    manifest_path = a.output/'run_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing experiment has another plan/code identity')
    dump_json(manifest_path, manifest)
    exp = Experiment(a)
    exp.warmup()
    if a.phase in ('all','diagnose'):
        gate = exp.diagnose()
        if not gate['passed']:
            exp.save('status.json', {'stage1': 'reader_gate_failed', 'stage2': 'not_run', 'gate': gate})
            return
        if a.phase == 'diagnose': return
    elif not json.loads((a.output/'oracle_summary.json').read_text())['gate']['passed']:
        raise ValueError('Reader gate did not pass')
    exp.setup_retriever()
    if a.phase in ('all','collect'):
        early = exp.collect(exp.stage1)
        summary = contrast_summary(early)
        exp.save('contrast_stage1.json', summary)
        if summary['contrast_questions'] < 5:
            exp.save('status.json', {'stage1': 'access_gate_failed_small', 'stage2': 'not_run', 'summary': summary})
            return
        rows = exp.collect(exp.train)
        summary = contrast_summary(rows)
        exp.save('contrast_train.json', summary)
        if summary['contrast_questions'] < 30 or summary['contrast_source_groups'] < 10:
            exp.save('status.json', {'stage1': 'access_gate_failed_expanded', 'stage2': 'not_run', 'summary': summary})
            return
        exp.save('status.json', {'stage1': 'passed', 'stage2': 'pending'})
        if a.phase == 'collect': return
    if a.phase in ('all','train'):
        from lsdrag.v11_training import train_heads
        rows = [json.loads((a.output/'train'/f'{q["qid"]}.json').read_text()) for q in exp.train]
        if contrast_summary(rows)['contrast_questions'] < 30:
            raise ValueError('Access gate failed')
        trained = train_heads(rows, a.output/'heads')
        exp.save('head_training.json', trained)
        if a.phase == 'train': return
    trained = json.loads((a.output/'head_training.json').read_text())
    exp.evaluate(trained['runs'])
    exp.save('status.json', {'stage1': 'completed', 'stage2': 'trained_and_development_evaluated',
             'note': 'controlled one-boundary continuation pilot, not a final test or ICML contribution proof'})


if __name__ == '__main__':
    main()
