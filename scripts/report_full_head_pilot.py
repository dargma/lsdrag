"""CPU-only progress report for fresh full-tool behavior-cloning experiments."""
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
import pickle
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def summarize_episodes(rows):
    return {'completed_questions':len(rows),
        'status_histogram':dict(collections.Counter(r.get('status','unknown') for r in rows)),
        'valid_answer_count':sum(bool(r.get('answer_format_valid')) for r in rows),
        'invalid_answer_count':sum(not bool(r.get('answer_format_valid')) for r in rows),
        'em':statistics.mean(r['em'] for r in rows) if rows and all('em' in r for r in rows) else None,
        'f1':statistics.mean(r['f1'] for r in rows) if rows and all('f1' in r for r in rows) else None,
        'questions_with_errors':sum(bool(r.get('errors')) for r in rows),
        'error_count':sum(len(r.get('errors',[])) for r in rows),
        'direct_format_error_count':sum('DirectOutputError:' in e.get('error','') for r in rows for e in r.get('errors',[])),
        'unique_delivered_images_sum':sum(r.get('unique_delivered_images',0) for r in rows),
        'image_delivered_question_count':sum(r.get('unique_delivered_images',0)>0 for r in rows),
        'stop_reason_histogram':dict(collections.Counter(str(r.get('stop_reason')) for r in rows))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run','data','report']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.report.mkdir(parents=True,exist_ok=True)
    manifest=read(args.run/'manifest.json')
    if not manifest:raise ValueError('Run manifest is required; no action contract can be inferred')
    contract=manifest['feature_contract'];actions=contract['actions']
    if len(actions)!=6 or len(set(actions))!=6:raise ValueError('Expected unique six-action manifest contract')
    dataset=read(args.data/'manifest.json',{})
    import torch
    torch.set_num_threads(2)
    records=[];episodes=[];artifacts=[];pending=[];issues=[];forced=0;hashes={}
    expected=manifest.get('train_question_order',[])
    for path in sorted((args.run/'teacher').glob('*.pt')):
        result_path=path.with_suffix('.json')
        if not result_path.exists():pending.append(path.stem);continue
        try:
            artifact=torch.load(path,map_location='cpu',weights_only=True)
            episode=read(result_path)
        except (EOFError,RuntimeError,ValueError,OSError,pickle.UnpicklingError) as error:
            issues.append({'qid':path.stem,'issue':'artifact_unreadable_possible_in_progress','type':type(error).__name__});continue
        if artifact.get('feature_contract')!=contract:
            issues.append({'qid':path.stem,'issue':'feature_contract_mismatch'});continue
        states=artifact['records'];features=artifact['features']
        if len(states)!=len(features) or features.ndim!=2:
            issues.append({'qid':path.stem,'issue':'feature_record_shape_mismatch'});continue
        invalid=[];valid=[]
        for state in states:
            if state.get('forced_final'):
                forced+=1;continue
            if state.get('qid')!=path.stem or state.get('selector')!='generated' or state.get('action') not in actions:
                invalid.append('identity_or_non_teacher_action');continue
            if state.get('index')!=actions.index(state['action']):invalid.append('action_index_mismatch');continue
            valid.append(state)
        if invalid:issues.append({'qid':path.stem,'issue':'invalid_state_records','count':len(invalid),'types':sorted(set(invalid))})
        records.extend(valid);episodes.append(episode);artifacts.append(path.stem)
        hashes[str(path.relative_to(args.run))]=sha(path);hashes[str(result_path.relative_to(args.run))]=sha(result_path)
    completed=set(artifacts)
    pending += [p.stem for p in (args.run/'teacher').glob('*.json') if p.stem not in completed]
    count=collections.Counter(r['action'] for r in records)
    coverage={action:len({r['qid'] for r in records if r['action']==action}) for action in actions}
    gate_file=args.run/'training_gate.json';training_file=args.run/'training.json'
    training=read(training_file)
    if training is None:training=read(args.run/'heads/training.json')
    training_gate=read(gate_file)
    phase='teacher_in_progress'
    if training_gate and not training_gate.get('passed'):phase='no_go_training_minimum'
    if training:
        phase='trained' if training.get('gate',{}).get('passed') else 'no_go_six_action_coverage'
    evaluation=read(args.run/'evaluation.json')
    if evaluation:phase='development_evaluated'
    elif any((args.run/'development').glob('*/*.json')):phase='development_in_progress'
    policies=collections.defaultdict(list)
    for path in sorted((args.run/'development').glob('*/*.json')):
        try:row=read(path)
        except (ValueError,OSError):continue
        policies[row.get('policy',path.parent.name)].append(row)
        hashes[str(path.relative_to(args.run))]=sha(path)
    seeds=manifest.get('seeds',[]);runs={r['seed']:r for r in (training or {}).get('runs',[])}
    seed_summary={str(seed):{'training_available':seed in runs,
        'checkpoint':runs.get(seed,{}).get('checkpoint'),
        'final_training_loss':runs[seed]['loss_curve'][-1] if seed in runs and runs[seed].get('loss_curve') else None,
        'development':summarize_episodes(policies[f'head_s{seed}']) if policies.get(f'head_s{seed}') else None,
        'heldout_teacher_agreement':(evaluation or {}).get('heldout_teacher_agreement',{}).get(str(seed))}
        for seed in seeds}
    failures=[]
    for path in sorted((args.run/'infrastructure_failures').glob('*.json')):
        row=read(path);failures.append({'qid':row.get('qid',path.stem),'status':row.get('status'),
            'errors':row.get('errors',[]),'scored_as_completed_teacher':path.stem in completed})
        hashes[str(path.relative_to(args.run))]=sha(path)
    warnings=[]
    if len(completed)<60:warnings.append('Fewer than60 completed teacher questions; initial20 is diagnostic, not a head-training success.')
    if any(v<5 for v in coverage.values()):warnings.append('One or more natural actions have fewer than5 distinct teacher questions; no six-action coverage success may be claimed.')
    if forced:warnings.append('Forced FINAL records were found and excluded; inspect upstream artifact integrity.')
    if issues:warnings.append('Artifact issues or currently incomplete writes are listed; no omitted artifact is treated as a successful episode.')
    if not evaluation:warnings.append('Held-out evaluation is incomplete or not yet run; no held-out success claim.')
    integrity=[]
    for file,digest in manifest.get('data_hashes',{}).items():
        path=Path(file)
        if not path.is_absolute():path=Path(__file__).resolve().parents[1]/path
        if not path.exists() or sha(path)!=digest:integrity.append(str(path))
    summary={'as_of_utc':datetime.now(timezone.utc).isoformat(),'phase':phase,
        'run':str(args.run.resolve()),'data':str(args.data.resolve()),'manifest_sha256':sha(args.run/'manifest.json'),
        'dataset_manifest_sha256':sha(args.data/'manifest.json') if (args.data/'manifest.json').exists() else None,
        'source_hashes':manifest.get('source_hashes',{}),'data_hashes':manifest.get('data_hashes',{}),
        'data_identity_mismatches':integrity,'feature_contract':contract,
        'teacher':summarize_episodes(episodes),'teacher_artifact_questions':len(completed),
        'completed_train_prefix':next((i for i,q in enumerate(expected) if q not in completed),len(expected)),
        'pending_or_unpaired_teacher_qids':sorted(set(pending)),
        'natural_state_count':len(records),'action_state_counts':{a:count[a] for a in actions},
        'action_distinct_question_coverage':coverage,'forced_final_records_excluded':forced,
        'forced_final_policy':'Forced FINAL is never a BC label; natural FINAL remains a genuine class.',
        'diagnostic_readiness_only':{'minimum_completed60':len(completed)>=60,
            'all_actions_minimum5_distinct_questions':all(v>=5 for v in coverage.values())},
        'training_gate':training_gate,'training_result_gate':(training or {}).get('gate'),
        'seeds':seed_summary,'development_policies':{p:summarize_episodes(rs) for p,rs in policies.items()},
        'evaluation':evaluation,'infrastructure_failures':failures,'artifact_issues':issues,'warnings':warnings,
        'data_scope':{k:dataset.get(k) for k in ['source','question_selection','train','development',
            'source_support_plus_table_overlap','prior100_overlap','not_a_final_test','no_unseen_source_claim']},
        'claims':{'objective':'Fresh generated-teacher behavior cloning; not utility learning or proof of improved decisions.',
            'world_model':False,'discarded_three_action_results_used':False,'latency_comparison':False,
            'lineage':'Public M3DocVQA/MMQA question lineage with fixed previously used288-document shared corpus.',
            'prior_exposure':'51 of full100 training questions overlap prior100; development0 prior100 qid overlap. Not strict unseen-source or final-test evaluation.'},
        'artifact_hashes':hashes,
        'report_source_sha256':sha(Path(__file__)),
        'plan_paths':[str(p) for p in [Path(__file__).resolve().parents[1]/'lab/registry'/args.run.name/'execution_plan.json',
                       Path(__file__).resolve().parents[1]/'docs/JEV_FULL_HEAD_PILOT_ASTRA_REVIEW.md',
                       args.data/'manifest.json',args.run/'manifest.json'] if p.exists()]}
    if integrity:summary['warnings'].append('Frozen data identity mismatches detected; interpret no result as a valid unchanged experiment until resolved.')
    (args.report/'full_head_progress.json').write_text(json.dumps(summary,indent=2))
    t=summary['teacher'];lines=['# Full six-action head pilot','',f"Status: **{phase}**. Snapshot: {summary['as_of_utc']}",'',
        'This is fresh-label behavior cloning of a generated teacher, with no world model. '
        'It does not demonstrate better decisions or answer quality. Discarded three-action results are not used. '
        'The GPU is shared, so this report makes no latency comparison.','',
        f"Completed teacher questions: **{len(completed)}**; natural labelled states: **{len(records)}**. "
        f"EM: {t['em']}; F1: {t['f1']}; valid answers: {t['valid_answer_count']}/{len(episodes)}.",'',
        f"Episode statuses: `{json.dumps(t['status_histogram'],sort_keys=True)}`. "
        f"Errors: {t['error_count']} across {t['questions_with_errors']} questions; selector/format errors: {t['direct_format_error_count']}. "
        f"Delivered images: {t['unique_delivered_images_sum']} episode-summed unique images across {t['image_delivered_question_count']} questions.",'',
        '## Natural action coverage','', '| Action | States | Distinct questions |','|---|---:|---:|']
    lines += [f'| {a} | {count[a]} | {coverage[a]} |' for a in actions]
    lines += ['',f'Forced FINAL records excluded: {forced}. Natural FINAL is retained. '
        'A20-question block is diagnostic. Training requires at least60 completed questions and the separate natural six-action coverage gate; a missing phase is not a passed gate.','',
        '## Training and held-out imitation','', '| Seed | Training available | Final train loss | Held-out agreement | Dev completed |',
        '|---|---|---:|---|---:|']
    for seed,row in seed_summary.items():
        agreement=row['heldout_teacher_agreement'];dev=row['development']
        lines.append(f"| {seed} | {row['training_available']} | {row['final_training_loss']} | "
            f"{agreement['accuracy'] if agreement else 'not available'} | {dev['completed_questions'] if dev else 0} |")
    lines += ['', 'All preregistered seeds are shown; no best seed is selected. Held-out teacher agreement measures imitation on the same generated-teacher states and is not task success.','',
        '## Scope and integrity','',summary['claims']['lineage'], '',summary['claims']['prior_exposure'],'',
        f"Run manifest SHA256: `{summary['manifest_sha256']}`. Data manifest SHA256: `{summary['dataset_manifest_sha256']}`. "
        'The JSON includes source/data hashes, artifact hashes, gates, phase summaries, incomplete records and provenance paths.','']
    lines += ['- '+w for w in summary['warnings']]
    (args.report/'FULL_HEAD_PROGRESS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'phase':phase,'teacher_questions':len(completed),'natural_states':len(records),
                      'warnings':summary['warnings'],'report':str(args.report.resolve())}))

if __name__=='__main__':main()
