"""Audit completed teacher records against frozen episodes and train-only gold."""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from common import ROOT, save, sha
from lsdrag.chunk_pointer import pointer_target, unread_candidates, visible_candidate_spans
from collect_decision_shared_teacher import document_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    teacher = args.teacher
    assert json.loads((teacher/'status.json').read_text())['phase'] == 'complete'
    spec = json.loads((teacher/'spec.json').read_text())
    split = json.loads(args.split.read_text())
    wanted = set(spec['qids'])
    assert len(wanted) == len(spec['qids']) == 576
    assert wanted == set(split['fit']) | set(split['validation'])
    assert not set(split['fit']) & set(split['validation'])
    episodes = {}
    for path in (teacher/'episodes').glob('*.json'):
        ep = json.loads(path.read_text())
        assert ep['qid'] not in episodes
        episodes[ep['qid']] = ep
    assert set(episodes) == wanted
    gold = {}
    for line in (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open():
        q = json.loads(line)
        if q['id'] in wanted:
            gold[q['id']] = {document_id(p) for p in q['paragraphs'] if p['is_supporting']}
    assert set(gold) == wanted
    records = torch.load(teacher/'teacher_states.pt', weights_only=True, map_location='cpu')
    tokenizer = AutoTokenizer.from_pretrained(spec['model'], local_files_only=True)
    seen = set()
    for r in records:
        key = (r['qid'], r['step_id'])
        assert key not in seen
        seen.add(key)
        ep = episodes[r['qid']]
        assert r['legal'][r['action']]
        history = ep['trajectory'][:r['step_id']]
        assert r['candidate_ids'] == [cid for cid, _ in unread_candidates(history)]
        ids = r['context_ids']
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        candidate_ids, spans = visible_candidate_spans(tokenizer, ids, history)
        assert candidate_ids == r['candidate_ids']
        assert [list(s) for s in spans] == [list(s) for s in r['candidate_spans']]
        assert r['candidate_features'].shape == (len(candidate_ids), 2560)
        assert torch.isfinite(r['candidate_features']).all() and torch.isfinite(r['pack']).all()
        if r['action'] == 2:
            actual_ids = ep['trajectory'][r['step_id']]['arguments']['chunk_ids'] if r['step_id'] < len(ep['trajectory']) else []
            assert actual_ids == r['teacher_chunk_ids']
            target, source = pointer_target(candidate_ids, actual_ids, gold[r['qid']]) if actual_ids else (None, 'invalid_teacher')
            assert target == r['pointer_target'] and source == r['pointer_source']
            assert r['gold_candidate_indices'] == [i for i, cid in enumerate(candidate_ids) if cid in gold[r['qid']]]
        else:
            assert r['pointer_target'] is None and r['pointer_source'] == 'not_read'
    expected = {(qid, i) for qid, ep in episodes.items() for i, d in enumerate(ep['decisions']) if not d['forced']}
    assert seen == expected
    valid = [r for r in records if r['pointer_target'] is not None]
    coverage = dict(fit_read_qids=len({r['qid'] for r in valid if r['qid'] in split['fit']}),
                    validation_read_qids_by_hop={h: len({r['qid'] for r in valid if r['qid'] in split['validation'] and r['qid'].startswith(h)}) for h in ('2hop', '3hop', '4hop')})
    coverage['passed'] = coverage['fit_read_qids'] >= 100 and min(coverage['validation_read_qids_by_hop'].values()) >= 10
    result = dict(audit_passed=True, episodes=len(episodes), records=len(records),
                  episodes_with_errors=sum(bool(ep['errors']) for ep in episodes.values()),
                  coverage=coverage, label_sources=dict(Counter(r['pointer_source'] for r in records)),
                  candidate_counts=dict(Counter(len(r['candidate_ids']) for r in valid)),
                  teacher_sha256=sha(teacher/'teacher_states.pt'), spec_sha256=sha(teacher/'spec.json'),
                  split_sha256=sha(args.split), source_sha256=sha(Path(__file__)))
    save(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
