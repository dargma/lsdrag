"""CPU-only paired report from existing acceleration development records."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    if not values:
        return None
    values = np.asarray(values, dtype=float)
    return {'mean':float(np.mean(values)), 'median':float(np.median(values)),
            'p95':float(np.quantile(values, .95))}


def cluster_ci(pairs, metric, repeats=2000):
    groups = collections.defaultdict(list)
    for original, fast in pairs:
        group = original.get('source_group')
        if group is None or fast.get('source_group') != group:
            return None
        groups[group].append(float(fast[metric])-float(original[metric]))
    if len(groups)<2:
        return None
    arrays = list(groups.values()); rng = np.random.default_rng(20261015)
    samples = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(arrays), len(arrays))
        samples.append(np.mean([v for i in chosen for v in arrays[i]]))
    return {'unit':'source_group cluster resampling; paired question mean fast-minus-original',
            'groups':len(groups), 'repeats':repeats, 'seed':20261015,
            'lower':float(np.quantile(samples,.025)), 'upper':float(np.quantile(samples,.975)),
            'interpretation':'descriptive pilot interval; not a noninferiority or final-test analysis'}


def compare(original, fast, expected):
    ids = sorted(original.keys() & fast.keys()); pairs = [(original[i],fast[i]) for i in ids]
    mismatches = {}
    for key in ['action','answers','f1','em']:
        mismatches[key] = [i for i in ids if original[i][key] != fast[i][key]]
    metrics = {}
    for metric in ['f1','em','total_seconds','decision_seconds']:
        old = [float(a[metric]) for a,b in pairs]; new = [float(b[metric]) for a,b in pairs]
        delta = [b-a for a,b in zip(old,new)]
        relative = [(b-a)/a*100 for a,b in zip(old,new) if a>0]
        mean_old = np.mean(old) if old else 0
        metrics[metric] = {'original':stats(old), 'fast':stats(new),
            'paired_delta_fast_minus_original':stats(delta),
            'per_pair_relative_delta_percent':stats(relative),
            'ratio_of_means_delta_percent':float((np.mean(new)/mean_old-1)*100) if mean_old else None,
            'paired_source_group_bootstrap_95':cluster_ci(pairs,metric)}
    return {'pairs':len(ids), 'expected_questions':expected, 'complete':len(ids)==expected,
            'original_records':len(original), 'fast_records':len(fast),
            'unpaired_original_qids':sorted(original.keys()-fast.keys()),
            'unpaired_fast_qids':sorted(fast.keys()-original.keys()),
            'disagreements':{k:{'n':len(v),'qids':v} for k,v in mismatches.items()},
            'strict_observed_equivalence':bool(ids) and not any(mismatches.values()),
            'metrics':metrics}


def num(value):
    return '—' if value is None else f'{value:.4f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--probe', type=Path)
    args = parser.parse_args(); args.report.mkdir(parents=True,exist_ok=True)
    amendment_path = args.run/'acceleration_amendment_manifest.json'
    amendment = json.loads(amendment_path.read_text()) if amendment_path.exists() else {}
    probe_path = args.probe or (Path(amendment['probe_path']) if amendment.get('probe_path') else None)
    probe = json.loads(probe_path.read_text()) if probe_path and probe_path.exists() else None
    by_policy = collections.defaultdict(dict); raw_hashes = {}
    for path in sorted((args.run/'development').glob('*/*.json')):
        row = json.loads(path.read_text()); name = row['policy']; qid = row['qid']
        if qid in by_policy[name]:
            raise ValueError('Duplicate policy/qid: '+name+'/'+qid)
        by_policy[name][qid] = row; raw_hashes[str(path.relative_to(args.run))] = digest(path)
    expected = 80
    pair_names = {'fast_first_token':'first_token'}
    for name in set(amendment.get('policies',[])) | set(by_policy):
        if name.startswith('fast_head_'): pair_names[name] = name[len('fast_'):]
    comparisons = {fast:{'original_policy':old, **compare(by_policy[old],by_policy[fast],expected)}
                   for fast,old in sorted(pair_names.items())}
    warnings = []
    if any(not c['complete'] for c in comparisons.values()):
        warnings.append('Incomplete development evaluation: report uses available matched pairs, expected80 per policy.')
    if len(comparisons)!=13:
        warnings.append(f'Found {len(comparisons)} paired contrasts; full grid requires13 (first-token plus12 heads).')
    if any(not c['strict_observed_equivalence'] for c in comparisons.values()):
        warnings.append('Some pairs disagree or are empty: do not describe the complete pipeline as equivalent.')
    if probe is None or probe.get('gate',{}).get('passed') is not True:
        warnings.append('A passing equivalence probe was not supplied/found.')
    if probe and amendment.get('probe_sha256') and digest(probe_path)!=amendment['probe_sha256']:
        warnings.append('Probe hash differs from the frozen amendment.')
    summary = {'run':str(args.run.resolve()), 'expected_questions_per_policy':expected,
        'paired_contrasts':comparisons, 'warnings':warnings,
        'interpretation':'Implementation acceleration should preserve decisions, not improve QA; observed QA differences are pipeline divergence, not an expected accuracy benefit.',
        'timing_delta_convention':'fast minus original: negative seconds/percent means faster',
        'amendment_sha256':digest(amendment_path) if amendment_path.exists() else None,
        'raw_record_hashes':raw_hashes,
        'probe':None if probe is None else {'path':str(probe_path.resolve()),'sha256':digest(probe_path),
            **{k:probe.get(k) for k in ['gate','feature_allclose_passed','action_disagreements',
                                       'comparisons','inputs','repeats','timings','limitation']}}}
    if probe:
        checks=[c for row in probe.get('rows',[]) for c in row.get('comparisons',{}).values()]
        summary['probe']['max_hidden_abs_delta']=max((c.get('feature_max_abs_delta',0) for c in checks),default=None)
        summary['probe']['max_selected_probability_abs_delta']=max((c.get('probs_max_abs_delta',0) for c in checks),default=None)
    (args.report/'acceleration_summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# JEV head acceleration: observed development results','',summary['interpretation'],'',
           'All deltas are fast minus original; negative latency deltas indicate a reduction. '
           'Matched question pairs only; no winning seed is selected. Source-group bootstrap intervals '
           'are descriptive and do not establish noninferiority.','']
    if warnings:
        lines += ['## Limitations','']+['- '+w for w in warnings]+['']
    lines += ['## Paired execution','',
        '| Fast policy | n /80 | action / answers / F1 mismatches | Δ F1 | total original → fast (s) | Δ total % | decision original → fast (s) | Δ decision % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,c in comparisons.items():
        m=c['metrics']; d=c['disagreements']
        mean=lambda metric,arm: m[metric][arm]['mean'] if m[metric][arm] else None
        lines.append(f"| {name} | {c['pairs']} | {d['action']['n']} / {d['answers']['n']} / {d['f1']['n']} | "
            f"{num(m['f1']['paired_delta_fast_minus_original']['mean'] if c['pairs'] else None)} | "
            f"{num(mean('total_seconds','original'))} → {num(mean('total_seconds','fast'))} | "
            f"{num(m['total_seconds']['ratio_of_means_delta_percent'])} | "
            f"{num(mean('decision_seconds','original'))} → {num(mean('decision_seconds','fast'))} | "
            f"{num(m['decision_seconds']['ratio_of_means_delta_percent'])} |")
    lines += ['','## Latency distributions','',
        '| Fast policy | metric | original median / p95 (s) | fast median / p95 (s) | paired Δ median / p95 (s) | mean Δ bootstrap95 (s) |',
        '|---|---|---:|---:|---:|---:|']
    for name,c in comparisons.items():
        for metric in ['total_seconds','decision_seconds']:
            m=c['metrics'][metric]; values=[]
            for arm in ['original','fast','paired_delta_fast_minus_original']:
                st=m[arm];values.append('—' if st is None else num(st['median'])+' / '+num(st['p95']))
            ci=m['paired_source_group_bootstrap_95'];interval='—' if ci is None else num(ci['lower'])+' to '+num(ci['upper'])
            lines.append(f"| {name} | {metric} | {' | '.join(values)} | {interval} |")
    lines += ['','## Backbone equivalence probe','']
    if probe:
        lines += [f"Probe passed: **{probe.get('gate',{}).get('passed')}**; comparisons: {probe.get('comparisons')}; "
            f"max hidden absolute delta: {summary['probe']['max_hidden_abs_delta']}; "
            f"max selected probability delta: {summary['probe']['max_selected_probability_abs_delta']}.",'',
            '```json',json.dumps({'gate':probe.get('gate'),'timings':probe.get('timings')},indent=2),'```','',
            probe.get('limitation','Finite probe agreement is not universal equivalence.')]
    else: lines += ['No probe results available.']
    lines += ['','The JSON report contains exact disagreement qids, raw-record SHA256 values, paired distributions, '
        'per-pair percentages, and source-group intervals. A finite probe and an80-question development run '
        'do not prove bitwise equivalence for every input, a new controller algorithm, or ICML-level accuracy improvement.','']
    (args.report/'HEAD_ACCELERATION_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'paired_contrasts':len(comparisons),'warnings':warnings,
                      'report':str(args.report.resolve())}))

if __name__=='__main__':
    main()
