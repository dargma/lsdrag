"""Paired evaluation audit; never pick a winner using held-out outcomes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2];ROOT=Path('/content/local_fast/lsdrag-text-v1')
def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();run=ROOT/'results'/args.run
 status=read(run/'status.json');assert status['phase']=='complete';spec=read(run/'spec.json');summary=read(run/'summary.json');eps=[read(p) for p in (run/'episodes').glob('*.json')];qids=spec['qids'];names=spec['candidates'];assert len(eps)==len(qids)*len(names)
 lookup={(e['policy'],e['qid']):e for e in eps};assert len(lookup)==len(eps)
 for name in names:
  es=[lookup[name,q] for q in qids]
  for metric in ['f1','generated_tokens','prompt_tokens','tool_calls','cost_units']:assert np.isclose(np.mean([e[metric] for e in es]),summary['policies'][name][metric])
 for e in eps:
  for d in e['decisions']:
   probs=np.array(d['policy_probs']);legal=np.array(d['legal']);assert abs(probs.sum()-1)<1e-5 and np.isfinite(probs).all() and (probs[~legal]==0).all()
 for metric,key in [('generated_tokens','generated'),('prompt_tokens','inputs')]:assert sum(e[metric] for e in eps)==status[key]
 rng=np.random.default_rng(20261046);draws=rng.integers(len(qids),size=(10000,len(qids)));comp={}
 base='grpo_baseline';assert base in names
 for name in names:
  if name==base:continue
  comp[name]={}
  for metric in ['f1','generated_tokens','prompt_tokens','tool_calls','cost_units','utility']:
   def val(e):return e['f1']-.03*e['cost_units'] if metric=='utility' else e[metric]
   a=np.array([val(lookup[name,q]) for q in qids]);b=np.array([val(lookup[base,q]) for q in qids]);delta=a-b
   comp[name][metric]={'delta':float(delta.mean()),'ci95':np.quantile(delta[draws].mean(1),[.025,.975]).tolist()}
   if metric=='generated_tokens':comp[name]['generated_ratio']={'ratio':float(a.mean()/b.mean()),'ci95':np.quantile(a[draws].mean(1)/b[draws].mean(1),[.025,.975]).tolist()}
  c=comp[name];c['joint_point_target']=bool(c['f1']['delta']>=.03 and c['generated_ratio']['ratio']<=.8);c['joint_directional_ci']=bool(c['f1']['ci95'][0]>0 and c['generated_ratio']['ci95'][1]<1);c['joint_target_ci']=bool(c['f1']['ci95'][0]>=.03 and c['generated_ratio']['ci95'][1]<=.8)
 result=summary|{'comparisons_vs_grpo':comp,'audit':{'episodes':len(eps),'unique_cells':len(lookup),'probability_and_cost_checks':True},'single_seed':True,'target':{'f1_delta':.03,'generated_ratio':.8},'ci_scope':'paired question bootstrap conditional on frozen policies; not training-seed population uncertainty'}
 out=REPO/'lab/reports'/args.run;out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 lines=['# Decision head 改善 결과', '',f"Split: {spec['split']}, n={len(qids)}, policies={len(names)}. 동일질문paired비교. 단일학습seed.",'','| 정책 | F1 | 생성token | 입력token | 도구호출 | 오류 |','|---|---:|---:|---:|---:|---:|']
 for name,v in summary['policies'].items():lines.append(f"| {name} | {100*v['f1']:.2f} | {v['generated_tokens']:.2f} | {v['prompt_tokens']:.1f} | {v['tool_calls']:.3f} | {v['errors']} |")
 lines+=['','## 고정 GRPO 대비','']
 for name,c in comp.items():lines += [f"- {name}: F1 delta {100*c['f1']['delta']:+.2f}pp, CI {c['f1']['ci95']}; 생성ratio {c['generated_ratio']}; point목표 {c['joint_point_target']}, 방향CI {c['joint_directional_ci']}, 목표CI {c['joint_target_ci']}."]
 lines+=['','F1+3%p 및생성token−20% 동시목표. 확률은action확률이며정답확률아님. 단일seed/조건부CI/공식dev-only아닌미사용train+dev원천holdout. 최종test결과로후보재선택금지. 전체학습/수집비용별도보고. 통제latency비교없음.']
 (out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(comp,indent=2))
if __name__=='__main__':main()
