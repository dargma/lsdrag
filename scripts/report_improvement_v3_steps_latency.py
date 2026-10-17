"""CPU-only aggregation of saved episode wall time and primitive interaction counts."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
RUN=Path('/content/local_fast/lsdrag-text-v1/results/improvement_v3_merged_20261017')
OUT=REPO/'lab/reports/text_integrated_20261016'
rows=[];provenance={}
for phase in ['validation','evaluation']:
    for p in sorted((RUN/phase).rglob('*.json')):
        e=json.loads(p.read_text());assert e['tool_calls']==len(e['trajectory'])
        turns=sum(c['kind'] in ['native_combined','decision'] for c in e['reader_calls'])+int(e['forced_final'])
        assert turns>=e['tool_calls'] and e['seconds']>0
        rows.append({'phase':phase,'qid':e['qid'],'policy':e['policy'],'tool_steps':e['tool_calls'],'interaction_turns':turns,'valid_final':int(e['answer_format_valid']),'errors':len(e['errors']),'latency_seconds':e['seconds'],'f1':e['f1'],'em':e['em']})
        provenance[str(p.relative_to(RUN))]=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def stats(rs):
    t=np.array([r['latency_seconds'] for r in rs]);return {'n':len(rs),'tool_steps_total':sum(r['tool_steps'] for r in rs),'tool_steps_mean':float(np.mean([r['tool_steps'] for r in rs])),'interaction_turns_total':sum(r['interaction_turns'] for r in rs),'interaction_turns_mean':float(np.mean([r['interaction_turns'] for r in rs])),'valid_finals':sum(r['valid_final'] for r in rs),'error_episodes':sum(r['errors']>0 for r in rs),'latency_total_seconds':float(t.sum()),'latency_mean_seconds':float(t.mean()),'latency_median_seconds':float(np.median(t)),'latency_p90_seconds':float(np.quantile(t,.9))}
groups={}
for phase in ['validation','evaluation']:
    groups[phase]={p:stats([r for r in rows if r['phase']==phase and r['policy']==p]) for p in sorted({r['policy'] for r in rows if r['phase']==phase})}
pairs=[]
for qid in sorted({r['qid'] for r in rows if r['phase']=='evaluation'}):
    es={r['policy']:r for r in rows if r['phase']=='evaluation' and r['qid']==qid};n=es['native'];s=es['always_generate']
    pairs.append({'qid':qid,'native':n,'primary':s,'tool_step_delta':s['tool_steps']-n['tool_steps'],'interaction_turn_delta':s['interaction_turns']-n['interaction_turns'],'latency_delta_seconds':s['latency_seconds']-n['latency_seconds'],'native_over_primary_time_ratio':n['latency_seconds']/s['latency_seconds']})
assert len(pairs)==20
rat=lambda ps:sum(p['native']['latency_seconds'] for p in ps)/sum(p['primary']['latency_seconds'] for p in ps)
paired={'n':20,'more_tool_steps':sum(p['tool_step_delta']>0 for p in pairs),'equal_tool_steps':sum(p['tool_step_delta']==0 for p in pairs),'fewer_tool_steps':sum(p['tool_step_delta']<0 for p in pairs),'observed_total_time_ratio_native_over_primary':rat(pairs),'mean_paired_latency_delta_seconds':float(np.mean([p['latency_delta_seconds'] for p in pairs]))}
same=[p for p in pairs if p['tool_step_delta']==0 and not p['native']['errors'] and not p['primary']['errors']]
paired['secondary_equal_tool_steps_no_errors']={'n':len(same),'observed_total_time_ratio':rat(same) if same else None,'warning':'Post-hoc subset; equal counts do not imply same states/actions/results. Not fixed-state causal timing.'}
report={'definitions':{'tool_steps':'Completed primitive tool executions; FINAL excluded.','interaction_turns':'Native combined action attempts or head decisions, plus forced FINAL if any. Includes terminal FINAL attempt or invalid decision; not number of model forwards.','latency_seconds':'run_episode wall time: prompt preparation, controller/prefill, generated or forced tokens, grammar, tool execution including semantic query encoding, FINAL and Python orchestration.','excluded':'Model loading, per-question corpus vector precomputation outside run_episode, training, scoring, file writes and backups. Full cold-start service latency was not instrumented.','measurement':'Single-run shared GPU observations. No controlled speedup claim; native error episode remains in primary totals.'},'groups':groups,'paired':paired,'pairs':pairs,'input_sha256':provenance,'new_gpu_run':False}
(OUT/'improvement_v3_steps_latency.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'improvement_v3_steps_latency.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Interaction steps와episode latency 추가 집계','', '기존56episode를CPU로집계했으며모델재실행없음. step은실제primitive tool call,interaction turn은FINAL/실패시도를포함한controller회수다. 두값모두model forward수와다르다.','', '| 구간 | 조건 | n | 평균 tool steps | 평균 interaction turns | 평균 latency(s) | 중앙값(s) | p90(s) | 합계(s) |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for phase,policies in groups.items():
 for p,v in policies.items():lines.append(f"| {phase} | {p} | {v['n']} | {v['tool_steps_mean']:.2f} | {v['interaction_turns_mean']:.2f} | {v['latency_mean_seconds']:.4f} | {v['latency_median_seconds']:.4f} | {v['latency_p90_seconds']:.4f} | {v['latency_total_seconds']:.4f} |")
lines+=['',f"fresh20에서primary tool steps 증가{paired['more_tool_steps']} /동일{paired['equal_tool_steps']} /감소{paired['fewer_tool_steps']}. 관측시간합계비 native/primary={paired['observed_total_time_ratio_native_over_primary']:.3f}. 통제된속도향상으로해석하지않는다.",'','latency는episode시작부터종료까지의wall time:컨트롤러·인자·답변·검색(query embedding포함)·Python처리포함. 모델load/사전corpus embedding/학습/채점/파일저장·백업제외. 따라서cold-start전체서비스latency는알수없다. 다른GPU작업간섭,조건별실제trajectory차이,순차실행·warmup/cache차이가있으며인과적속도비교아님. native형식실패1건도기본통계에포함.','',f"동일tool step수·양쪽오류없는사후subset n={len(same)}의시간비={rat(same):.3f}. 같은step수라도같은state/도구결과는아니므로보조기술통계만이다.",'','원문항별수치CSV와JSON을함께저장했다. 원본episode/동결source는변경하지않았다.']
(OUT/'improvement_v3_steps_latency.md').write_text('\n'.join(lines)+'\n')
p=OUT/'improvement_v3_final.json';v=json.loads(p.read_text());v['interaction_and_latency']={'report':'lab/reports/text_integrated_20261016/improvement_v3_steps_latency.json','groups':groups,'paired':paired,'controlled_speed_claim':False};p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
p=OUT/'improvement_v3_final.md';s=p.read_text();header='\n## Interaction steps·latency 추가\n';s=s.split(header)[0]+header+'\n'+ '\n'.join(lines[4:])+'\n';p.write_text(s)
print(json.dumps({'groups':groups,'paired':paired},indent=2))
