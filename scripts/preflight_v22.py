"""CPU-only restoration, split, and answer-cap audit; never opens test answers."""
import json
import math
import sys
from collections import Counter
from pathlib import Path

from common import ROOT, REPO, CONFIG, save, sha


def main():
    out = ROOT / 'specs/v22'
    split = json.loads((REPO / 'configs/split_mixhop.json').read_text())
    ids = [q for k in ('train', 'validation', 'test') for q in split[k]]
    assert len(ids) == len(set(ids))
    wanted = set(split['train'])
    raw = {}
    for line in (ROOT / 'source/musique/musique_ans_v1.0_train.jsonl').open():
        # Only train questions are retained or analyzed.
        q = json.loads(line)
        if q['id'] in wanted:
            raw[q['id']] = q
    assert set(raw) == wanted
    sys.path.insert(0, str(ROOT / 'source/musique_scorer'))
    from metrics.answer import normalize_answer
    import hashlib
    fit, validation = [], []
    for hop in ('2hop', '3hop', '4hop'):
        group = sorted((q for q in wanted if q.startswith(hop)),
                       key=lambda q: hashlib.sha256(f'20261110:{q}'.encode()).hexdigest())
        assert len(group) == 192
        fit += group[:153]
        validation += group[153:]
    def family(q):
        return set(q.split('__', 1)[1].split('_'))
    fit_atoms = set().union(*(family(q) for q in fit))
    overlap = [q for q in validation if family(q) & fit_atoms]
    lengths = {}
    aliases = {}
    for qid, q in raw.items():
        a = [q['answer']] + q['answer_aliases']
        aliases[qid] = [{'raw': len(x), 'normalized': len(normalize_answer(x))} for x in a]
        lengths[qid] = min(x['normalized'] for x in aliases[qid] if x['normalized'])
    ordered = sorted(lengths.values())
    cap = max(60, ordered[math.ceil(.99 * len(ordered)) - 1])
    blocked = {q: n for q, n in lengths.items() if n > cap}
    corpus = Path(CONFIG['corpus'])
    cm = json.loads((corpus / 'manifest.json').read_text())
    for filename, key in [('documents.jsonl', 'document_sha256'),
                          ('vectors.npy', 'vectors_sha256'), ('lexical.sqlite', 'lexical_sha256')]:
        assert sha(corpus / filename) == cm[key], filename
    for part, rec in cm['sources'].items():
        assert sha(ROOT / f'source/musique/musique_ans_v1.0_{part}.jsonl') == rec['sha256']
    result = dict(seed=20261110, fit=fit, internal_validation=validation,
                  train_smoke16=fit[:6] + fit[153:158] + fit[306:311],
                  pilot24=fit[:8] + fit[153:161] + fit[306:314],
                  split_sha256=sha(REPO / 'configs/split_mixhop.json'),
                  answer_cap_candidate=cap, all_aliases_blocked=blocked,
                  answer_lengths=aliases, internal_validation_family_overlap=overlap,
                  family_overlap_note='qid-disjoint; shared atomic subquestions are audited, not claimed family-disjoint',
                  corpus=cm, test_answers_used=False)
    save(out / 'preflight.json', result)
    save(out / 'kd_split.json', dict(fit=fit, validation=validation, seed=20261110,
                                   source_split_sha256=result['split_sha256']))
    for name, qids in [('legacy_smoke16', result['train_smoke16']),
                       ('legacy_dev192', split['validation'])]:
        spec = dict(run_name='v22_' + name, split='train' if 'smoke' in name else 'validation',
                    reader_profile='fullft', model=CONFIG['model'], corpus=CONFIG['corpus'], qids=qids,
                    document_sha256=cm['document_sha256'], vectors_sha256=cm['vectors_sha256'],
                    source_sha256={}, policies={'native': {'kind': 'native'}},
                    max_seconds=7200, max_generated=1800000, max_inputs=60000000,
                    original_query_diagnostic=False)
        save(out / f'{name}.json', spec)
    print(json.dumps(dict(train=len(fit), internal_validation=len(validation),
                          family_overlap_qids=len(overlap), cap=cap, blocked=blocked,
                          assets_verified=True, path=str(out)), indent=2))


if __name__ == '__main__':
    main()
