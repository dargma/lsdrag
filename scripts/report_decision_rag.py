"""Read-only paired diagnosis; support-document hits are proxies, not sufficiency."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(row, gold, chunks, figures):
    costs = Counter()
    for call in row['reader_calls']:
        u = call['usage']
        if 'decision_usage' in u:
            for k in ('prompt_tokens', 'generated_tokens', 'prefill_seconds', 'decode_seconds', 'head_seconds'):
                costs['decision_'+k] += u['decision_usage'].get(k, 0)
            category = 'final' if u.get('selected_action') == 'FINAL' else 'arguments'
            for k in ('prompt_tokens', 'generated_tokens', 'prefill_seconds', 'decode_seconds'):
                costs[category+'_'+k] += u.get('generation_usage', {}).get(k, 0)
        else:
            for k in ('prompt_tokens', 'generated_tokens', 'prefill_seconds', 'decode_seconds'):
                costs['native_combined_'+k] += u.get(k, 0)
    ids = set(row.get('read_chunk_ids', []))
    read_docs = set()
    for identity in ids:
        c = chunks.get(identity, {})
        read_docs.update(c.get('document_ids', [c.get('document_id')]))
    delivered_docs = set()
    for identity in row.get('delivered_figure_ids', []):
        f = figures.get(identity, {})
        delivered_docs.update(f.get('document_ids', [f.get('document_id')]))
    retrieved_docs = set()
    for t in row['trajectory']:
        if t['tool_name'].endswith('search') and isinstance(t['result'], list):
            for c in t['result']:
                if isinstance(c, dict):
                    source = chunks.get(c.get('chunk_id'), c)
                    retrieved_docs.update(source.get('document_ids', [source.get('document_id')]))
    support = set(gold.get('support_document_ids', []))
    tool_errors = [t['result']['error'] for t in row['trajectory'] if isinstance(t['result'], dict) and 'error' in t['result']]
    requests = [t['tool_name']+json.dumps(t['arguments'], sort_keys=True) for t in row['trajectory']]
    observations = []
    if not row['answer_format_valid']: observations.append('answer_format_invalid')
    if tool_errors: observations.append('tool_error_observed')
    if row['stop_reason'] == 'max_loops': observations.append('iteration_budget_reached')
    if row['f1'] < 1:
        observations.append('nonperfect_answer')
        if support and not (support & (retrieved_docs | read_docs)):
            observations.append('no_annotated_support_document_encountered')
        elif support and support <= read_docs:
            observations.append('all_support_documents_read_but_nonperfect_answer_NOT_proof_of_sufficient_evidence')
        else: observations.append('partial_support_document_access_or_unknown')
    return {'qid':row['qid'], 'policy':row['policy'], 'em':row['em'], 'f1':row['f1'],
        'answer_format_valid':row['answer_format_valid'], 'status':row['status'],
        'stop_reason':row['stop_reason'], 'tool_calls':len(row['trajectory']),
        'tool_distribution':dict(Counter(t['tool_name'] for t in row['trajectory'])),
        'repeated_exact_requests':len(requests)-len(set(requests)),
        'read_chunks':len(ids), 'delivered_images':row['unique_delivered_images'],
        'prompt_tokens':row['prompt_tokens'], 'completion_tokens':row['completion_tokens'],
        'observed_wall_seconds':row['seconds'], 'cost_components':dict(costs),
        'reader_errors':row['errors'], 'tool_errors':tool_errors, 'observations':observations,
        'support_document_count':len(support), 'support_documents_read':len(support & read_docs),
        'support_documents_retrieved':len(support & retrieved_docs),
        'support_documents_with_delivered_images':len(support & delivered_docs),
        'question':row['question'], 'predicted_answers':row['answers'], 'reference_answers':gold['answers']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('/content/local_fast/lsdrag-head-v12/data'))
    p.add_argument('--corpus',type=Path,default=Path('/content/local_fast/lsdrag-work/data/ir_final288'))
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--limit',type=int,default=20)
    a=p.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text())
    qs=read_jsonl(a.data/'train_questions.jsonl')[:a.limit]
    gold={q['qid']:q for q in read_jsonl(a.data/'gold.jsonl')}
    chunks={r['chunk_id']:r for r in read_jsonl(a.corpus/'chunks.jsonl')}
    figures={r['figure_id']:r for r in read_jsonl(a.corpus/'figures.jsonl')}
    policies=['native','generated','first_token'];rows=[];missing=[];hashes={}
    for q in qs:
        for name in policies:
            path=a.run/'diagnostic'/name/(q['qid']+'.json')
            if not path.exists(): missing.append({'qid':q['qid'],'policy':name});continue
            row=json.loads(path.read_text())
            if row['qid']!=q['qid'] or row['policy']!=name:raise ValueError('Wrong episode identity')
            if row['tool_contract_sha256']!=manifest['tool_contract_sha256']:raise ValueError('Tool contract mismatch')
            rows.append(analyze(row,gold[q['qid']],chunks,figures))
            hashes[str(path.relative_to(a.run))]=hashlib.sha256(path.read_bytes()).hexdigest()
    summary={}
    for policy in policies:
        chosen=[r for r in rows if r['policy']==policy]
        if not chosen:continue
        summary[policy]={'n':len(chosen), **{k:statistics.mean(r[k] for r in chosen) for k in ['em','f1','tool_calls','prompt_tokens','completion_tokens','delivered_images','observed_wall_seconds']},
            'answer_format_valid':sum(r['answer_format_valid'] for r in chosen),
            'observation_counts':dict(Counter(o for r in chosen for o in r['observations'])),
            'cost_component_totals':dict(sum((Counter(r['cost_components']) for r in chosen),Counter()))}
    paired={}
    bykey={(r['qid'],r['policy']):r for r in rows}
    for left,right in [('first_token','generated'),('first_token','native')]:
        pairs=[(bykey[(q['qid'],left)],bykey[(q['qid'],right)]) for q in qs if (q['qid'],left) in bykey and (q['qid'],right) in bykey]
        paired[left+'_minus_'+right]={'n':len(pairs),'mean_f1_difference':statistics.mean(x['f1']-y['f1'] for x,y in pairs) if pairs else None,
            'f1_wins':sum(x['f1']>y['f1'] for x,y in pairs),'f1_losses':sum(x['f1']<y['f1'] for x,y in pairs)}
    result={'scope':'train diagnostic; no learned-head/test/latency superiority claim', 'expected_questions':a.limit,
        'complete_requested_block':not missing, 'stage1_all20_complete':a.limit==20 and not missing,
        'missing':missing,'policies':summary,'paired':paired,'episodes':rows,'artifact_sha256':hashes,
        'limitations':['Support-document hits do not prove evidence sufficiency or correct image use.',
        'Native jointly generates tool names/arguments; its generation cost cannot be split.',
        'Observed wall time is uncontrolled; same-call output budget is not equal total compute.',
        'Observations are diagnostics, not established causal failure labels.']}
    a.report.mkdir(parents=True,exist_ok=True)
    (a.report/'diagnostic.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# Stage1 diagnostic','',result['scope'],'',f"Complete: {not missing}; expected {a.limit} questions × 3 policies.",'', '| Policy | n | EM | F1 | Mean calls | Mean prompt tokens | Mean completion tokens |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in summary.items():lines.append(f"| {name} | {r['n']} | {r['em']:.4f} | {r['f1']:.4f} | {r['tool_calls']:.2f} | {r['prompt_tokens']:.1f} | {r['completion_tokens']:.1f} |")
    lines+=['']+result['limitations']
    (a.report/'DIAGNOSTIC.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete':not missing,'policies':summary,'paired':paired},ensure_ascii=False))

if __name__=='__main__':main()
