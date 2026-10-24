"""Rescore existing H3/H1/GRPO episodes; no model execution or selection."""
import hashlib
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[2]
def read(path):
    return json.loads(path.read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    h = ROOT/'results/decision_budget_stop_h3_validation64_20261024'
    g = ROOT/'results/decision_shared_validation64_20261023'
    hs, gs = read(h/'spec.json'), read(g/'spec.json')
    for key in ['qids', 'model', 'corpus', 'document_sha256', 'vectors_sha256']:
        assert hs[key] == gs[key], key
    assert len(hs['qids']) == len(set(hs['qids'])) == 64
    for item in [*hs['policies'].values(), gs['policies']['shared_grpo']]:
        checkpoint = Path(item['checkpoint'])
        if checkpoint.exists():
            digest = sha(checkpoint)
        else:
            archive = read(REPO/'lab/registry'/checkpoint.parent.name/'archive.json')
            assert sha(Path(archive['path'])) == archive['sha256']
            with tarfile.open(archive['path']) as tf:
                members = [m for m in tf.getmembers() if Path(m.name).name == checkpoint.name]
                assert len(members) == 1
                digest = hashlib.sha256(tf.extractfile(members[0]).read()).hexdigest()
        assert digest == item['sha256']
    selection = read(REPO/'lab/registry/decision_shared_final_20261023/selection_plan.json')
    assert not set(hs['qids']) & set(selection['test_qids'] + selection['reserved_qids'])
    gold = {}
    for split in ['train', 'dev']:
        for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
            q = json.loads(line)
            if q['id'] in hs['qids']:
                gold[q['id']] = q
    sys.path.insert(0, str(ROOT/'source/musique_scorer'))
    from metrics.answer import compute_f1
    data, hashes = {}, {}
    for name, run, prefix in [('h3', h, 'budget_stop'), ('h1', h, 'selected_reference'), ('grpo', g, 'shared_grpo')]:
        rows = []
        for qid in hs['qids']:
            p = run/'episodes'/f'{prefix}_{qid}.json'
            e = read(p); q = gold[qid]
            assert e['qid'] == qid
            assert e['f1'] == max(compute_f1(a, e['answer']) for a in [q['answer']] + q['answer_aliases'])
            assert e['generated_tokens'] == sum(c.get('generated_tokens', 0) for c in e['reader_calls'])
            hashes[str(p)] = sha(p)
            rows.append(e)
        data[name] = rows
    metrics = {k: {field: sum(e[field] for e in rows)/64 for field in ['f1', 'generated_tokens']} for k, rows in data.items()}
    comparisons = {}
    for ref in ['grpo', 'h1']:
        groups = defaultdict(lambda: {'questions': 0, 'saved_tokens': 0})
        for a, b in zip(data[ref], data['h3']):
            group = 'quality_gain' if b['f1'] > a['f1'] else 'quality_loss' if b['f1'] < a['f1'] else 'positive_retained' if b['f1'] > 0 else 'both_zero'
            groups[group]['questions'] += 1
            groups[group]['saved_tokens'] += a['generated_tokens'] - b['generated_tokens']
        comparisons[ref] = dict(groups=groups, f1_delta=metrics['h3']['f1']-metrics[ref]['f1'], generated_ratio=metrics['h3']['generated_tokens']/metrics[ref]['generated_tokens'])
    report = dict(verified=True, scope='Repeatedly exposed development64; grouped net savings, not independent confirmation or causal mechanism proof', metrics=metrics, comparisons=comparisons, episode_sha256=hashes, errors={k:sum(bool(e['errors']) for e in rows) for k,rows in data.items()})
    out = REPO/'lab/reports/decision_h3_role_audit_20261024'
    out.mkdir(parents=True, exist_ok=True)
    (out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='episode_sha256'}, indent=2))
if __name__ == '__main__':
    main()
