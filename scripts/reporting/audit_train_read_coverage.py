"""Train-only representational coverage; oracle membership is not learned accuracy."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from lsdrag.text_integrated import options_for, target_slot

ROOT = Path('/content/local_fast/lsdrag-text-v1')
REPO = Path(__file__).resolve().parents[2]

def main():
    gold = {q['id']: q for q in map(json.loads, (ROOT / 'source/musique/musique_ans_v1.0_train.jsonl').open())}
    paths = sorted((ROOT / 'results/decision_progress_stop120_train_eval_20261024/episodes').glob('h3_*.json'))
    assert len(paths) == 120
    counts = Counter(); tokens = Counter(); rows = []; hashes = {}
    for path in paths:
        raw = path.read_bytes(); hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        e = json.loads(raw); assert e['qid'] in gold
        calls = [c for c in e['reader_calls'] if 'raw_json' in c]
        assert len(calls) == len(e['decisions'])
        assert sum(c['generated_tokens'] for c in calls) == e['generated_tokens']
        history = []
        for d, call in zip(e['decisions'], calls):
            if d['action'] == 3:
                continue
            step = e['trajectory'][len(history)]
            if step['tool_name'] == 'read_chunk':
                ids = step['arguments']['chunk_ids']; groups = []
                for prior in reversed(history):
                    result = prior['result']
                    group = list(dict.fromkeys(r['chunk_id'] for r in result if isinstance(r, dict) and isinstance(r.get('chunk_id'), str))) if isinstance(result, list) else []
                    if group: groups.append(group)
                observed = list(dict.fromkeys(x for g in groups for x in g))
                candidates = [g[:8] for g in groups] + [[x] for x in observed]
                flags = dict(all_reads=True, single=len(ids)==1,
                    old_four=target_slot(options_for(gold[e['qid']]['question'], history), 'read_chunk', step['arguments'])>0,
                    uncapped_groups_singletons=ids in candidates,
                    ordered_observed_pointer=all(x in observed for x in ids))
                for key, value in flags.items():
                    if value: counts[key] += 1; tokens[key] += call['generated_tokens']
                rows.append(dict(qid=e['qid'], turn=len(history), ids_count=len(ids), observed_count=len(observed), generated_tokens=call['generated_tokens'], **flags))
            history.append(step)
        assert len(history) == len(e['trajectory'])
    report = dict(scope='120 train-only H3 reference episodes. Representation upper bounds, NOT selection accuracy, H2 coverage, or measured token savings.', questions=len(paths), read_questions=len({r['qid'] for r in rows}), calls=dict(counts), decode_tokens=dict(tokens), rows=rows, episode_sha256=hashes)
    dest = REPO / 'lab/reports/decision_train_read_coverage_20261024'; dest.mkdir(parents=True, exist_ok=True)
    (dest / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','episode_sha256']}, indent=2))

if __name__ == '__main__': main()
