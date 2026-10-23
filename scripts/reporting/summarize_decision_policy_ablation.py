"""Audit fresh projection evaluation and avoid equating non-significance with equality."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path('/content/local_fast/lsdrag-text-v1/results/decision_policy_ablation_20261023');REPO=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def main():
 status=read(ROOT/'status.json');assert status['phase']=='complete';m=read(ROOT/'manifest.json');summary=read(ROOT/'summary.json');eps=[read(p) for p in (ROOT/'episodes').glob('*.json')];assert len(eps)==360;decisions=0
 for e in eps:
  for d in e['decisions']:
   assert abs(sum(d['policy_probs'])-1)<1e-5;assert all(p==0 for p,l in zip(d['policy_probs'],d['legal']) if not l);decisions+=1
 for name,values in summary['policies'].items():
  rows=[e for e in eps if e['policy']==name];assert len(rows)==40 and {e['qid'] for e in rows}==set(m['test'])
  for k in ['f1','cost_units','tool_calls','generated_tokens','prompt_tokens']:assert np.isclose(values[k],np.mean([e[k] for e in rows]))
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in m['input_sha256'].items())
 avgs={}
 for mode in ['full','global_bias','stop_only','tool_only']:
  avgs[mode]={k:float(np.mean([summary['policies'][f'{mode}_s{s}'][k] for s in [1,2]])) for k in ['f1','utility','tool_calls','generated_tokens','prompt_tokens']}
 gate=all(summary['full_minus']['global_bias'][k]['ci95_conditional_questions'][0]>0 for k in ['f1','utility'])
 result=summary|{'seed_average':avgs,'state_specific_increment_gate_passed':gate,'audit':{'episodes':360,'decisions':decisions,'inputs_unchanged':True},'scope':'Frozen policy projections, not a test of semantic understanding or equal-budget training algorithms.'};out=REPO/'lab/reports/decision_policy_ablation_20261023';out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 lines=['# PPO 전역 편향·종료·도구 선택 분해 결과','', '새family-disjoint40문항×9조건=360회. 두학습head의평균을같은질문끼리비교. 기존teacher/평가문항으로편향을튜닝하지않았으며편향은원train67상태PPO확률을근사한다.','', '| 정책 | F1 | utility | 호출 | 생성token |','|---|---:|---:|---:|---:|']
 for mode,v in {'anchor':summary['policies']['anchor'],**avgs}.items():lines.append(f"| {mode} | {v['f1']:.4f} | {v['utility']:.4f} | {v['tool_calls']:.3f} | {v['generated_tokens']:.2f} |")
 lines+=['','## Paired comparisons','']
 for mode,c in summary['full_minus'].items():
  for metric,v in c.items():lines.append(f"- full−{mode} {metric}: {v['mean']:+.4f}, 95%CI {v['ci95_conditional_questions']}; seed별차이 {v['seed_deltas']}.")
 lines+=['',f'State-specific incremental contribution gate: {gate}. CI가0포함하면차이미확정이며동등성을증명하지않는다.','', '## 해석 범위','', 'Globalbias도기존anchor의state-dependent판단은유지한다. 제거한것은PPO추가수정의state의존성이다. STOP/toolprojection은각현재state에서해당margin/argmax를보존하지만경로가달라져후속state도달라질수있다. 오류/강제FINAL과seed별수치는JSON에보존.','', '전체head가우월해도semantic이해를직접입증하지않는다. 충분/부족/무관근거를같은질문에서통제하는후속검증이필요하다. 평균편향근사오차와한정된traincoverage때문에globalbias가약해진가능성도고려한다. 동일budgetreward재학습비교가아니다.','', f'편향fitting진단: {summary["fit"]}', '',f'원본360개/결정{decisions}확률·합법성/입력hash검증통과. 통제latency없음.','', '[동결 protocol](JEV_DECISION_POLICY_ABLATION_20261023.md)']
 (REPO/'docs/JEV_DECISION_POLICY_ABLATION_REPORT_20261023.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'average':avgs,'full_minus':summary['full_minus'],'gate':gate,'audit':result['audit']},indent=2))
if __name__=='__main__':main()
