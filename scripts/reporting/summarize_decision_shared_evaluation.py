"""Paired evaluation audit; never pick a winner using held-out outcomes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2];ROOT=Path('/content/local_fast/lsdrag-text-v1')
def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--baseline',default='shared_grpo');args=ap.parse_args();run=ROOT/'results'/args.run
 status=read(run/'status.json');assert status['phase']=='complete';spec=read(run/'spec.json');summary=read(run/'summary.json');eps=[read(p) for p in (run/'episodes').glob('*.json')];qids=spec['qids'];names=list(spec['policies']);assert len(eps)==len(qids)*len(names)
 lookup={(e['policy'],e['qid']):e for e in eps};assert len(lookup)==len(eps)
 for name in names:
  es=[lookup[name,q] for q in qids]
  for metric in ['f1','generated_tokens','prompt_tokens','tool_calls','cost_units']:assert np.isclose(np.mean([e[metric] for e in es]),summary['policies'][name][metric])
 for e in eps:
  for d in e['decisions']:
   if spec['policies'][e['policy']]['kind']=='native':
    assert d['policy_probs'] is None;continue
   probs=np.array(d['policy_probs']);legal=np.array(d['legal']);assert abs(probs.sum()-1)<1e-5 and np.isfinite(probs).all() and (probs[~legal]==0).all()
 for metric,key in [('generated_tokens','generated'),('prompt_tokens','inputs')]:assert sum(e[metric] for e in eps)==status[key]
 for filename,digest in spec['source_sha256'].items():assert hashlib.sha256((REPO/filename).read_bytes()).hexdigest()==digest,filename
 for desc in spec['policies'].values():
  if 'checkpoint' in desc:assert hashlib.sha256(Path(desc['checkpoint']).read_bytes()).hexdigest()==desc['sha256']
 split=read(ROOT/'results/decision_state_policy_iteration_20261023/manifest.json')
 assert set(qids)<=set(split[spec['split']])
 if spec['split']=='test':assert qids==split['test']
 rng=np.random.default_rng(20261046);draws=rng.integers(len(qids),size=(10000,len(qids)));comp={}
 base=args.baseline;assert base in names
 for name in names:
  if name==base:continue
  comp[name]={}
  for metric in ['f1','generated_tokens','prompt_tokens','tool_calls','cost_units','utility']:
   def val(e):return e['f1']-.03*e['cost_units'] if metric=='utility' else e[metric]
   a=np.array([val(lookup[name,q]) for q in qids]);b=np.array([val(lookup[base,q]) for q in qids]);delta=a-b
   comp[name][metric]={'delta':float(delta.mean()),'ci95':np.quantile(delta[draws].mean(1),[.025,.975]).tolist()}
   if metric=='generated_tokens':comp[name]['generated_ratio']={'ratio':float(a.mean()/b.mean()),'ci95':np.quantile(a[draws].mean(1)/b[draws].mean(1),[.025,.975]).tolist()}
  c=comp[name];c['joint_point_target']=bool(c['f1']['delta']>=.05 and c['generated_ratio']['ratio']<=.7);c['joint_directional_ci']=bool(c['f1']['ci95'][0]>0 and c['generated_ratio']['ci95'][1]<1);c['joint_target_ci']=bool(c['f1']['ci95'][0]>=.05 and c['generated_ratio']['ci95'][1]<=.7)
 result=summary|{'baseline':base,'comparisons_vs_baseline':comp,'audit':{'episodes':len(eps),'unique_cells':len(lookup),'probability_and_cost_checks':True},'single_seed':True,'target':{'f1_delta':.05,'generated_ratio':.7},'ci_scope':'paired question bootstrap conditional on frozen policies; not training-seed population uncertainty'}
 if spec['split']=='validation':
  old=set(read(ROOT/'results/decision_state_outcomes_20261023/manifest.json')['validation'])
  groups={'previously_exposed':[q for q in qids if q in old],'new_validation':[q for q in qids if q not in old]}
  result['validation_subgroups']={}
  for group,ids in groups.items():
   if not ids:continue
   means={name:{metric:float(np.mean([lookup[name,q][metric] for q in ids])) for metric in ['f1','generated_tokens','prompt_tokens','tool_calls']} for name in names}
   result['validation_subgroups'][group]={'n':len(ids),'policies':means,'note':'Descriptive subgroup means, no candidate selection on final test.'}
 out=REPO/'lab/reports'/args.run;out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 lines=['# Decision head 개선 결과', '',f"Split: {spec['split']}, n={len(qids)}, policies={len(names)}. 동일질문paired비교. 단일학습seed.",'','| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |','|---|---:|---:|---:|---:|---:|']
 for name,v in summary['policies'].items():lines.append(f"| {name} | {100*v['f1']:.2f} | {v['generated_tokens']:.2f} | {v['prompt_tokens']:.1f} | {v['tool_calls']:.3f} | {v['errors']} |")
 lines+=['','## 지정 baseline 대비','']
 for name,c in comp.items():lines += [f"- {name}: F1 delta {100*c['f1']['delta']:+.2f}pp, CI {c['f1']['ci95']}; 생성ratio {c['generated_ratio']}; point목표 {c['joint_point_target']}, 방향CI {c['joint_directional_ci']}, 목표CI {c['joint_target_ci']}."]
 lines+=['','F1+5%p 및생성token−30% 동시목표. 확률은action확률이며정답확률아님. 단일seed/조건부CI/공식train원천의학습/검증가족분리holdout이며공통101962문단검색환경. validation은후보선택자료이므로최종확증아님. 최종test결과로후보재선택금지. 전체학습/수집비용별도보고. 통제latency비교없음.']
 if 'validation_subgroups' in result:
  lines+=['','## Validation exposure subgroups','','```json',json.dumps(result['validation_subgroups'],ensure_ascii=False,indent=2),'```']
 if spec['split']=='test':
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(9,6))
  for name,c in comp.items():
   x=100*(1-c['generated_ratio']['ratio']);y=100*c['f1']['delta'];xc=c['generated_ratio']['ci95'];yc=c['f1']['ci95']
   ax.errorbar(x,y,xerr=[[max(0,x-100*(1-xc[1]))],[max(0,100*(1-xc[0])-x)]],yerr=[[max(0,y-100*yc[0])],[max(0,100*yc[1]-y)]],fmt='o',capsize=3,label=name)
  ax.axvline(30,color='green',linestyle='--',label='Joint target: 30% tokens, +5pp F1');ax.axhline(5,color='green',linestyle='--');ax.axvline(0,color='gray',linewidth=.5);ax.axhline(0,color='gray',linewidth=.5)
  ax.set_xlabel('Generated token reduction vs baseline (%)');ax.set_ylabel('F1 difference vs baseline (percentage points)');ax.set_title('Shared-corpus paired evaluation; marginal 95% bootstrap intervals');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'tradeoff.svg');plt.close(fig)
  lines+=['','![Paired tradeoff](tradeoff.svg)']
 (out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(comp,indent=2))
if __name__=='__main__':main()
