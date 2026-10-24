"""Report correctness-stratified token accounting from verified raw episodes."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source/musique_scorer'))
from metrics.answer import compute_exact, compute_f1


def summarize(rows):
    n = len(rows)
    total = sum(r['generated_tokens'] for r in rows)
    return dict(questions=n, total_generated=total,
                mean_generated=total/n if n else None)


def main():
    reportdir = REPO/'lab/reports/decision_h3_role_audit_20261024'
    audit = json.loads((reportdir/'audit.json').read_text())
    assert audit['verified']
    gold = {}
    for split in ['train', 'dev']:
        for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
            q = json.loads(line)
            gold[q['id']] = [q['answer']] + q['answer_aliases']
    policies = defaultdict(dict)
    labels = {'budget_stop':'h3', 'selected_reference':'h1', 'shared_grpo':'grpo'}
    for path, expected in audit['episode_sha256'].items():
        raw = Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        e = json.loads(raw)
        e['em'] = max(compute_exact(a, e['answer']) for a in gold[e['qid']])
        assert e['f1'] == max(compute_f1(a, e['answer']) for a in gold[e['qid']])
        assert e['generated_tokens'] == sum(c.get('generated_tokens',0) for c in e['reader_calls'])
        policies[labels[e['policy']]][e['qid']] = e
    per_policy = {}
    for name, indexed in policies.items():
        rows = list(indexed.values())
        assert len(rows) == 64
        bands = {
            'correct_em1':[r for r in rows if r['em']==1],
            'incorrect_em0':[r for r in rows if r['em']==0],
            'incorrect_partial_f1':[r for r in rows if r['em']==0 and 0<r['f1']<1],
            'incorrect_zero_f1':[r for r in rows if r['em']==0 and r['f1']==0],
            'incorrect_full_f1_nonexact':[r for r in rows if r['em']==0 and r['f1']==1],
        }
        assert sum(len(bands[k]) for k in ['incorrect_partial_f1','incorrect_zero_f1','incorrect_full_f1_nonexact']) == len(bands['incorrect_em0'])
        assert sum(sum(r['generated_tokens'] for r in bands[k]) for k in ['correct_em1','incorrect_em0']) == sum(r['generated_tokens'] for r in rows)
        per_policy[name] = dict(all=summarize(rows), groups={k:summarize(v) for k,v in bands.items()})
    paired = {}
    for ref in ['grpo','h1']:
        groups = {}
        assert set(policies[ref]) == set(policies['h3'])
        for re, he, name in [(1,1,'both_correct'),(0,1,'h3_only_correct'),(1,0,'reference_only_correct'),(0,0,'both_incorrect')]:
            ids = [q for q in policies[ref] if policies[ref][q]['em']==re and policies['h3'][q]['em']==he]
            a = summarize([policies[ref][q] for q in ids])
            b = summarize([policies['h3'][q] for q in ids])
            groups[name] = dict(questions=len(ids),reference=a,h3=b,saved_tokens=a['total_generated']-b['total_generated'])
        assert sum(g['questions'] for g in groups.values()) == 64
        paired[ref] = groups
    result = dict(verified=True, correctness='Official MuSiQue normalized exact match over answer and aliases; partial F1 remains incorrect and is reported separately', scope='Repeated development64; per-policy conditional means compare different question subsets; use paired groups for savings attribution. No new inference.', per_policy=per_policy, paired=paired)
    (reportdir/'correctness_cost.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
