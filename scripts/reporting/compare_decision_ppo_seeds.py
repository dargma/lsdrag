"""Two training seeds, shared40 test questions: paired question-level aggregation."""
from pathlib import Path
import json,numpy as np
REPO=Path(__file__).resolve().parents[2];BASE=Path('/content/local_fast/lsdrag-text-v1/results')
def read(p):return json.loads(p.read_text())
def main():
 names=['decision_ppo_tempered_20261023','decision_ppo_tempered_seed2_20261023'];runs=[BASE/n for n in names];ms=[read(r/'manifest.json') for r in runs];assert ms[0]['test']==ms[1]['test'];assert all(read(r/'status.json')['phase']=='complete' for r in runs);qids=ms[0]['test'];policies=ms[0]['policies'];summaries=[read(r/'summary.json') for r in runs];records={}
 for ri,run in enumerate(runs):
  for name in policies:
   for q in qids:records[ri,name,q]=read(run/f'episodes/test_{name}_{q}.json')
 rng=np.random.default_rng(20261035);idx=rng.integers(40,size=(4000,40));comparisons={}
 for comparator in ['tempered_anchor','historical_bc','rwr']:
  comparisons[comparator]={}
  for metric in ['f1','utility','tool_calls','generated_tokens']:
   def val(e):return e['f1']-.03*e['cost_units'] if metric=='utility' else e[metric]
   deltas=np.array([[val(records[ri,'ppo',q])-val(records[ri,comparator,q]) for q in qids] for ri in range(2)]);avg=deltas.mean(0)
   comparisons[comparator][metric]={'mean':float(avg.mean()),'seed_means':deltas.mean(1).tolist(),'conditional_question_ci95':np.quantile(avg[idx].mean(1),[.025,.975]).tolist()}
 avg={name:{k:float(np.mean([s['policies'][name][k] for s in summaries])) for k in ['f1','utility','tool_calls','generated_tokens','prompt_tokens']} for name in policies}
 result={'training_seeds':[m['seed'] for m in ms],'independent_test_questions':40,'episodes':608,'policies_seed_average':avg,'ppo_minus':comparisons,'seed_results':summaries,'limitation':'Only two training seeds; CI resamples shared questions after seed averaging, does not estimate population training-seed uncertainty.'}
 target=REPO/'lab/reports/decision_ppo_two_seeds_20261023';target.mkdir(parents=True,exist_ok=True);(target/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Tempered PPO 두 학습 seed 종합','', '독립on-policy수집/update seed2개,동일fresh40질문. 총608episodes. 평가80개독립질문이라고세지않는다. 두번째seed는첫결과열람전설정동결.','', '| 정책 | 평균F1 | 평균utility | 호출 | 생성token |','|---|---:|---:|---:|---:|']
 for name,v in avg.items():lines.append(f"| {name} | {v['f1']:.4f} | {v['utility']:.4f} | {v['tool_calls']:.3f} | {v['generated_tokens']:.2f} |")
 lines+=['','## Seed별 PPO−anchor','']
 for metric in ['f1','utility']:
  c=comparisons['tempered_anchor'][metric];lines.append(f"- {metric}: seed차이 {c['seed_means']}; seed평균 {c['mean']:+.4f}, 질문조건부95%CI {c['conditional_question_ci95']}.")
 lines+=['','## PPO−RWR','']
 for metric in ['f1','utility']:
  c=comparisons['rwr'][metric];lines.append(f"- {metric}: {c['mean']:+.4f}, 질문조건부95%CI {c['conditional_question_ci95']}.")
 lines+=['','이전Q회귀probe와이번PPO는초기화·MLP표현·온라인수집데이터·학습목표가다르므로회귀실패를RL만으로해결했다고해석하지않는다. 행동개선은미래성공확률calibration이나state-dependent전략의입증과도다르다.','', 'PPO와RWR동일배치이나PPO수집정책을사용하므로독립on-policy알고리즘우월성비교가아니다. 두seed만으로학습seed모집단변동을추정할수없고CI는고정두seed의질문변동만반영. 품질·도구호출·입력/생성token은따로판단;통제latency/생성token절감/최신방법우월/novelty주장없음.','', '[Seed1](JEV_DECISION_PPO_TEMPERED_REPORT_20261023.md) · [Seed2](JEV_DECISION_PPO_TEMPERED_SEED2_REPORT_20261023.md) · [동결설계](JEV_DECISION_PPO_TEMPERED_20261023.md)']
 (REPO/'docs/JEV_DECISION_PPO_TWO_SEEDS_REPORT_20261023.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'policies':avg,'ppo_minus':comparisons},indent=2))
if __name__=='__main__':main()
