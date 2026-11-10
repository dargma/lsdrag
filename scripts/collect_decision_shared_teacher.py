"""Train-only native trajectory collection in the frozen shared corpus."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from common import ROOT, REPO, SPLIT, save, sha, load_head
from lsdrag.decision_policy import rollout
from lsdrag.decision_policy_recovery import RecoveryPolicyReader
from lsdrag.text_decision_benchmark import TextCorpus
from lsdrag.v11_embedding_worker import V11ProcessEmbedder

def read(path):
    return json.loads(path.read_text())

def document_id(paragraph):
    identity = json.dumps([paragraph['title'], paragraph['paragraph_text']], ensure_ascii=False, separators=(',', ':'))
    return 'c'+hashlib.sha256(identity.encode()).hexdigest()[:24]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--limit', type=int, default=0, help='bounded smoke; resume same frozen spec afterwards')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    spec = read(Path(args.spec))
    assert spec['split'] == 'train' and set(spec['policies']) == {'native_teacher'}
    v22 = spec.get('profile') == 'v22'
    split_contract = read(REPO/'configs/split_mixhop.json') if v22 else SPLIT
    assert set(spec['qids']) <= set(split_contract['train'])
    if not v22:
        assert spec['qids'] == SPLIT['train']
    assert not set(spec['qids']) & set(split_contract['validation']+split_contract['test'])
    pool = Path(spec['corpus'])
    manifest = read(pool/'manifest.json')
    assert manifest['embeddings_complete']
    assert sha(pool/'documents.jsonl') == spec['document_sha256'] == manifest['document_sha256']
    assert sha(pool/'vectors.npy') == spec['vectors_sha256'] == manifest['vectors_sha256']
    for filename, digest in spec['source_sha256'].items():
        assert sha(REPO/filename) == digest, filename
    lock = (ROOT/'decision_policy.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = ROOT/'results'/spec['run_name']
    out.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        assert read(out/'spec.json') == spec, 'Cannot resume with a changed data/source contract'
    save(out/'spec.json', spec)
    start = time.monotonic()
    counts = dict(episodes=0, generated=0, inputs=0)
    def status(phase, **kw):
        save(out/'status.json', dict(phase=phase, pid=None if phase in ['complete', 'failed', 'smoke_complete'] else os.getpid(), seconds=time.monotonic()-start, **counts, **kw))
    status('loading')
    torch.set_num_threads(2)
    torch.manual_seed(spec.get('seed', 20261051))
    embed = None
    try:
        rows = [json.loads(line) for line in (pool/'documents.jsonl').open()]
        vectors = np.load(pool/'vectors.npy', mmap_mode='r')
        assert vectors.shape == (len(rows), 768)
        assert all(set(r)=={'chunk_id', 'title', 'text'} for r in rows)
        raw = {}
        for split in ['train', 'dev']:
            for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
                q = json.loads(line)
                if q['id'] in spec['qids']:
                    raw[q['id']] = q
        assert set(raw) == set(spec['qids'])
        sys.path.insert(0, str(ROOT/'source/musique_scorer'))
        from metrics.answer import compute_f1
        heads = {}
        for name, desc in spec['policies'].items():
            if desc['kind'] == 'native':
                heads[name] = None
            else:
                assert sha(Path(desc['checkpoint'])) == desc['sha256']
                heads[name] = load_head(desc, 'cuda')
        reader_class = RecoveryPolicyReader
        if v22:
            from lsdrag.fullft import SampledReader
            reader_class = SampledReader
        reader = reader_class(spec['model'], SimpleNamespace(max_output_tokens=512, context_limit=16384))
        if v22:
            reader.v22 = True
            reader.answer_max_chars = spec['answer_max_chars']
        embed = V11ProcessEmbedder()
        corpus = TextCorpus(rows, vectors, embed)
        prompt = (REPO/'prompts/text_rag_integrated_v5.txt').read_text()
        deadline = time.perf_counter()+spec['max_seconds']
        episodes = []
        training_states = []
        diagnostics = []
        if args.resume:
            episodes = [read(p) for p in sorted((out/'episodes').glob('*.json'))]
            training_states = torch.load(out/'teacher_states.pt', weights_only=True)
            diagnostics = read(out/'retrieval_diagnostics.json')
            counts.update(episodes=len(episodes), generated=sum(e['generated_tokens'] for e in episodes),
                          inputs=sum(e['prompt_tokens'] for e in episodes))
        completed_qids = {e['qid'] for e in episodes}
        collected = 0
        for qi, qid in enumerate(spec['qids']):
            if qid in completed_qids:
                continue
            if args.limit and collected >= args.limit:
                break
            q = raw[qid]
            # Gold is scorer-only; rollout sees question and the identical full corpus.
            gold = {document_id(p) for p in q['paragraphs'] if p['is_supporting']}
            assert gold <= corpus.by_id.keys()
            found = corpus.call('semantic_search', dict(query=q['question'], top_k=5)) if spec.get('original_query_diagnostic', True) else []
            hit = {r['chunk_id'] for r in found}
            diagnostics.append(dict(qid=qid, support_recall_at5=len(gold & hit)/len(gold), all_support_at5=gold <= hit, retrieved_ids=sorted(hit)))
            for name, head in heads.items():
                if hasattr(head, 'reset_episode'):
                    head.reset_episode()
                assert counts['generated']+4608 < spec['max_generated']
                assert counts['inputs']+222048 < spec['max_inputs']
                native = spec['policies'][name]['kind'] == 'native'
                ep, steps, _ = rollout(reader, q['question'], rows, vectors, embed, prompt, head, deadline, stochastic=False, epsilon=0., native_fast=native)
                for si, step in enumerate(steps):
                    if not step['forced']:
                        assert step['legal'][step['action']]
                        record = {k:step[k] for k in ['pack','legal','action']} | {'qid':qid, 'episode_f1':max(compute_f1(a, ep['answer']) for a in [q['answer']]+q['answer_aliases'])}
                        if v22:
                            from lsdrag.chunk_pointer import pointer_target
                            record.update(step['capture'], step_id=si, pointer_target=None,
                                          pointer_source='not_read', teacher_chunk_ids=[], gold_candidate_indices=[])
                            if step['action'] == 2:
                                teacher_ids = ep['trajectory'][si]['arguments']['chunk_ids'] if si < len(ep['trajectory']) else []
                                if teacher_ids:
                                    target, source = pointer_target(record['candidate_ids'], teacher_ids, gold)
                                else:
                                    target, source = None, 'invalid_teacher'
                                record.update(pointer_target=target, pointer_source=source, teacher_chunk_ids=teacher_ids,
                                              gold_candidate_indices=[i for i, cid in enumerate(record['candidate_ids']) if cid in gold])
                        training_states.append(record)
                # Atomic checkpoint; used to resume the bounded smoke without repeating episodes.
                torch.save(training_states, out/'teacher_states.pt.tmp')
                (out/'teacher_states.pt.tmp').replace(out/'teacher_states.pt')
                if native:
                    # rollout's dummy uniform distribution is not the native model's policy.
                    for decision in ep['decisions']:
                        decision['policy_probs'] = None
                        decision['old_logp'] = None
                observed, read_ids = set(), set()
                for step in ep['trajectory']:
                    ids = {r['chunk_id'] for r in step['result'] if isinstance(r, dict) and 'chunk_id' in r}
                    observed |= ids
                    if step['tool_name'] == 'read_chunk':
                        read_ids |= ids
                ep.update(qid=qid, policy=name, f1=max(compute_f1(a, ep['answer']) for a in [q['answer']]+q['answer_aliases']), retrieved_support_recall=len(gold & observed)/len(gold), all_support_retrieved=gold <= observed, read_support_recall=len(gold & read_ids)/len(gold))
                save(out/'episodes'/f'{name}_{qid}.json', ep)
                episodes.append(ep)
                counts['episodes'] += 1
                counts['generated'] += ep['generated_tokens']
                counts['inputs'] += ep['prompt_tokens']
                status('evaluation', questions_started=qi+1, last_policy=name)
            save(out/'retrieval_diagnostics.json', diagnostics)
            collected += 1
        policies = {}
        for name in heads:
            subset = [e for e in episodes if e['policy']==name]
            policies[name] = {k:float(np.mean([e[k] for e in subset])) for k in ['f1', 'generated_tokens', 'prompt_tokens', 'tool_calls', 'cost_units', 'retrieved_support_recall', 'all_support_retrieved', 'read_support_recall']}
            policies[name]['errors'] = sum(bool(e['errors']) for e in subset)
        summary = dict(policies=policies, n=len(spec['qids']), split=spec['split'], counts=counts, corpus_documents=len(rows), original_question_retrieval={k:float(np.mean([r[k] for r in diagnostics])) for k in ['support_recall_at5', 'all_support_at5']}, controlled_latency=False, note='Existing heads are transfer controls, not shared-corpus retrained policies. Native-fast generated action has additional accounted generation/prefill costs; dummy policy probabilities removed.')
        summary.update(teacher_states=len(training_states), teacher_states_sha256=sha(out/'teacher_states.pt'), note='Native action hard labels only, no dummy uniform policy/logprob used as learning targets. All non-forced states retained, including failed episodes; QAs are train-only.')
        save(out/'summary.json', summary)
        if v22:
            from collections import Counter
            valid = [r for r in training_states if r.get('pointer_target') is not None]
            save(out/'pointer_coverage.json', dict(valid_read_states=len(valid),
                 valid_read_qids=len({r['qid'] for r in valid}),
                 label_sources=dict(Counter(r.get('pointer_source') for r in training_states)),
                 candidate_counts=dict(Counter(len(r['candidate_ids']) for r in valid)),
                 collected_questions=len(episodes), expected_questions=len(spec['qids'])))
        status('complete' if len(episodes) == len(spec['qids']) else 'smoke_complete')
        print(json.dumps(summary, indent=2))
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise
    finally:
        if embed:
            embed.close()

if __name__ == '__main__':
    main()
