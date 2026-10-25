"""POC P2: error attribution and oracle-stop input headroom from saved dev64 episodes (CPU, gold used offline only)."""
import argparse
import json
from pathlib import Path

ROOT = Path('/content/local_fast/lsdrag-text-v1')


def per_decision_prompts(ep):
    """Group reader usage by decision: head decisions log [capture, continue], native_fast logs [capture, plan, continue]."""
    calls, out, i = ep['reader_calls'], [], 0
    for d in ep['decisions']:
        width = 3 if d.get('policy_probs') is None and not d['forced'] else 2
        out.append(sum(u.get('prompt_tokens', 0) for u in calls[i:i+width]))
        i += width
    assert i == len(calls), (i, len(calls))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--policy', default='before')
    args = parser.parse_args()
    run = Path(args.run)
    spec = json.loads((run/'spec.json').read_text())
    raw = {}
    for line in (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open():
        q = json.loads(line)
        if q['id'] in spec['qids']:
            raw[q['id']] = q
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_decision_shared_corpus import document_id
    rows = []
    for qid in spec['qids']:
        ep = json.loads((run/'episodes'/f'{args.policy}_{qid}.json').read_text())
        gold = {document_id(p) for p in raw[qid]['paragraphs'] if p['is_supporting']}
        seen, first_all = set(), None
        for k, step in enumerate(ep['trajectory']):
            seen |= {r['chunk_id'] for r in step['result'] if isinstance(r, dict) and 'chunk_id' in r}
            if first_all is None and gold <= seen:
                first_all = k+1  # tool calls needed before all support was observed
        prompts = per_decision_prompts(ep)
        oracle_inputs = sum(prompts[:first_all+1]) if first_all is not None else sum(prompts)
        correct = ep['f1'] == 1.0 or ep.get('em', 0) == 1
        if correct:
            bucket = 'correct'
        elif ep['forced_final']:
            bucket = 'forced_at_cap'
        elif ep['all_support_retrieved']:
            bucket = 'retrieved_but_wrong'
        else:
            bucket = 'support_not_retrieved'
        rows.append(dict(qid=qid, f1=ep['f1'], bucket=bucket, tool_calls=ep['tool_calls'], first_all_support_call=first_all,
                         prompt_tokens=ep['prompt_tokens'], oracle_stop_prompt_tokens=oracle_inputs,
                         retrieved_support_recall=ep['retrieved_support_recall'], read_support_recall=ep['read_support_recall']))
    n = len(rows)
    buckets = {}
    for r in rows:
        buckets[r['bucket']] = buckets.get(r['bucket'], 0)+1
    total_in = sum(r['prompt_tokens'] for r in rows)
    oracle_in = sum(r['oracle_stop_prompt_tokens'] for r in rows)
    summary = dict(policy=args.policy, n=n, buckets=buckets,
                   mean_f1=sum(r['f1'] for r in rows)/n,
                   all_support_observed=sum(r['first_all_support_call'] is not None for r in rows),
                   mean_prompt_tokens=total_in/n, mean_oracle_stop_prompt_tokens=oracle_in/n,
                   oracle_stop_input_saving=1-oracle_in/total_in,
                   note='Oracle stop = FINAL right after all gold support first appears in observations; uses gold offline, upper bound only. Correct = F1 1.0.')
    (run/f'error_attribution_{args.policy}.json').write_text(json.dumps(dict(summary=summary, rows=rows), indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
