"""Auditable run-level token ledger; pending runs never count as completed."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path('/content/local_fast/lsdrag-text-v1')
REPO=Path(__file__).resolve().parents[2]
GROUPS={
    'historical_grpo_reference':['decision_grpo96_20261023'],
    'restricted_pool_exploration':[
        'decision_state_outcomes_20261023','decision_state_validation1_20261023',
        'decision_state_validation2_20261023','decision_state_validation3_20261023',
        'decision_grpo_continuation_20261023','decision_state_outcomes_seed2_20261023',
        'decision_state_validation_seed2_20261023','decision_state_policy_iteration_20261023'],
    'shared_corpus_restart':[
        'decision_shared_calibration16_20261023','decision_shared_teacher192_20261023',
        'decision_shared_initialization_validation16_20261023','decision_shared_grpo192_20261023',
        'decision_shared_outcomes192_20261023','decision_shared_validation64_20261023',
        'decision_shared_structure_validation64_20261023','decision_shared_test80_20261023'],
    'selected_head_grpo_continuation':['decision_selected_head_grpo192_20261024','decision_selected_head_grpo192_20261024_r1','decision_selected_head_grpo_validation64_20261024_r1'],
    'adaptive_policy_followup':['decision_selected_head_grpo192_20261024_zero_gate','decision_selected_head_grpo_validation64_20261024_zero_gate','decision_opportunity_value_validation64_20261024','decision_constrained_policy_validation64_20261024','decision_current_policy_refresh48_20261024','decision_quality_head_grpo192_20261024','decision_quality_head_grpo_validation64_20261024','decision_pareto_refresh_validation64_20261024','decision_context_pareto_validation64_20261024']}

def main():
    GROUPS['offline_structure_and_reward']=[
        'decision_reward_replay_validation64_20261024',
        'decision_structure_four_validation64_20261024',
        'decision_structure_attention_validation64_20261024',
        'decision_frozen_adapters_validation64_20261024']
    rows=[]
    for group,names in GROUPS.items():
        for name in names:
            run=ROOT/'results'/name;path=run/'status.json'
            if not path.exists():
                rows.append(dict(group=group,run=name,phase='not_started',counts=None))
                continue
            status=json.loads(path.read_text())
            interruption=run/'interruption.json'
            if interruption.exists():
                audit=json.loads(interruption.read_text())
                status={**status,**{k:audit[k] for k in ['episodes','generated','inputs']},'phase':'interrupted'}
            counts={k:status.get(k,0) for k in ['episodes','generated','inputs']}
            verified=False
            if status['phase'] in ('complete','failed','interrupted'):
                eps=[json.loads(p.read_text()) for p in (run/'episodes').glob('*.json')]
                assert len(eps)==counts['episodes'],name
                assert sum(e['generated_tokens'] for e in eps)==counts['generated'],name
                assert sum(e['prompt_tokens'] for e in eps)==counts['inputs'],name
                verified=True
            rows.append(dict(group=group,run=name,phase=status['phase'],counts=counts,seconds=status.get('seconds'),complete_raw_totals_verified=verified and status['phase']=='complete',terminal_raw_totals_verified=verified,status_path=str(path)))
    totals={}
    for group in GROUPS:
        completed=[r for r in rows if r['group']==group and r['phase']=='complete']
        totals[group]={k:sum(r['counts'][k] for r in completed) for k in ['episodes','generated','inputs']}
        failed=[r for r in rows if r['group']==group and r['phase']=='failed']
        totals[group]['failed_run_costs']={k:sum(r['counts'][k] for r in failed) for k in ['episodes','generated','inputs']}
        totals[group]['failed_runs']=[r['run'] for r in failed]
        interrupted=[r for r in rows if r['group']==group and r['phase']=='interrupted']
        totals[group]['interrupted_run_costs']={k:sum(r['counts'][k] for r in interrupted) for k in ['episodes','generated','inputs']}
        totals[group]['completed_runs']=len(completed)
        totals[group]['uncompleted_runs']=[r['run'] for r in rows if r['group']==group and r['phase']!='complete']
    cpu_path=ROOT/'results/decision_opportunity_value_20261024/status.json'
    cpu_fit=json.loads(cpu_path.read_text()) if cpu_path.exists() else {'phase':'not_started'}
    constrained_path=ROOT/'results/decision_constrained_policy_20261024/status.json'
    constrained_fit=json.loads(constrained_path.read_text()) if constrained_path.exists() else {'phase':'not_started'}
    result=dict(constrained_cache_fitting={'run':'decision_constrained_policy_20261024','status':constrained_fit,'new_llm_rollouts':0},cpu_cache_fitting={'run':'decision_opportunity_value_20261024','status':cpu_fit,'new_llm_rollouts':0,'reuses':'decision_shared_outcomes192_20261023 and decision_shared_teacher192_20261023'},snapshot_utc=datetime.now(timezone.utc).isoformat(),runs=rows,completed_totals=totals,scope='Named decision-policy campaign only; historical PPO/preparatory runs outside this list are excluded. Shared-corpus and restricted-pool results are separate experimental settings.',limitations=['Running counts are status snapshots, not certified terminal totals. Pending runs contribute no invented cost.','CPU cache fitting and corpus embedding/indexing are not LLM generated tokens and are not included in these token totals.','Wall times may include loading/waiting and shared-GPU effects; not controlled comparative latency.','Historical GRPO reference training is separate from new exploration cost; no claim every listed run occurred inside original eight-hour window.'])
    out=REPO/'lab/reports/decision_campaign_budget_20261023';out.mkdir(parents=True,exist_ok=True)
    (out/'ledger.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Decision-policy experiment budget','','Completed and failed-run totals are checked against raw episodes. Failed costs remain separate in ledger.json and are not discarded. Running values are provisional status snapshots.','','|Setting|Run|Status|Episodes|Generated tokens|Input tokens|','|---|---|---|---:|---:|---:|']
    for row in rows:
        c=row['counts'] or {};lines.append(f"|{row['group']}|{row['run']}|{row['phase']}|{c.get('episodes','—')}|{c.get('generated','—')}|{c.get('inputs','—')}|")
    lines+=['','## Scope','',result['scope'],'',*['- '+s for s in result['limitations']]]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(totals,indent=2))

if __name__=='__main__':main()
