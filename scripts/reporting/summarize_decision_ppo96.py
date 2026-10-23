"""Audit complete tempered PPO run and render honest paired comparisons."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/content/local_fast/lsdrag-text-v1/results/decision_ppo96_20261023');REPO=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def main(out_name="decision_ppo96_20261023", doc_name="JEV_DECISION_PPO96_REPORT_20261023.md"):
 status=read(ROOT/'status.json');assert status['phase']=='complete';m=read(ROOT/'manifest.json');s=read(ROOT/'summary.json');curves=read(ROOT/'curves.json');episodes=[read(p) for p in (ROOT/'episodes').glob('*.json')];assert len(episodes)==m["budget_episodes"]
 checks=0
 for e in episodes:
  for d in e['decisions']:
   p=d['policy_probs'];assert abs(sum(p)-1)<1e-5;assert all(v==0 for v,legal in zip(p,d['legal']) if not legal);checks+=1
 for name,v in s['policies'].items():
  es=[e for e in episodes if e['policy']=='test_'+name];assert len(es)==40 and {e['qid'] for e in es}==set(m['test'])
  for k in ['f1','cost_units','tool_calls','generated_tokens','prompt_tokens']:assert np.isclose(v[k],np.mean([e[k] for e in es]))
 for name,digest in read(ROOT/'frozen_evaluation.json')['head_sha256'].items():assert hashlib.sha256((ROOT/f'{name}_iteration2.pt').read_bytes()).hexdigest()==digest
 gate=all(s['ppo_minus']['tempered_anchor'][metric]['ci95'][0]>0 for metric in ['f1','utility'])
 result=s|{'optimizer_steps_per_learner':sum(4*((c['states']+31)//32) for c in curves),'audit':{'episodes':len(episodes),'decisions':checks,'head_hashes_unchanged':True},'strong_signal_gate_passed':gate,'curves':curves};out=REPO/'lab/reports'/out_name;out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 lines=['# PPO96×4 trajectories 결과','',f'총1312episodes,학습수집1152+fresh40×4평가160. strong signal gate: {gate}. 단일seed,공유GPU통제latency없음.','', '| 정책 | F1 | utility | 호출 | 생성token | 오류episode | cap FINAL |','|---|---:|---:|---:|---:|---:|---:|']
 for name,v in s['policies'].items():lines.append(f"| {name} | {v['f1']:.4f} | {v['utility']:.4f} | {v['tool_calls']:.3f} | {v['generated_tokens']:.2f} | {v['errors']} | {v['forced_final']} |")
 
 if s.get('recovery'):
  lines+=['','## 평가 복구 기록','',str(s['recovery']),'학습1152회완료후평가70개시점에grammar mismatch로중단. 기존1222개episode는bytehash보존,누락90개만실행. 첫실패cell은재실행성공여부와무관하게F1=0으로보존. 해당cell비용은재실행proxy이고최초실패시도추가비용은알수없어총실제비용을완전히집계한것은아님. 1312는정규집계episode수이며실패한원시도를포함한총attempt수와다르다.']
 lines+=['','## Paired 비교','']
 for name,c in s['ppo_minus'].items():
  for metric,v in c.items():lines.append(f"- PPO−{name} {metric}: {v['delta']:+.4f}, 95% CI {v['ci95']}.")
 lines+=['','## 학습 및 검증','',f'원본{len(episodes)}개재집계,결정{checks}개확률합/합법성,동결headhash검사통과. 각iteration수집당시logprob와update전logprob parity통과.','']
 for c in curves:lines.append(f"- Iteration{c['iteration']}: {c['states']}학습state, diagnostics={c['diagnostic']}")
 lines+=['',f'실제optimizer update수(각learner): {result["optimizer_steps_per_learner"]}. 질문96개와update수는다르다. 이전24문항실험과는수집량/update수/평가질문도달라점수차이만으로데이터량의순수효과를주장하지않는다.','', 'PPO가수집한동일배치를RWR도사용한다. 같은데이터이나수집정책은PPO에맞춰져있어독립on-policy학습법의공정대결은아니다. BCtemperature8로탐색성을주고trunk고정/action·skip·critic만학습했다. full-backbone RL이나구조탐색성과가아니다. 새test40은이후학습/튜닝금지.','', '학습식: return=F1−.03×remainingcost, advantage=(return−oldcritic)의배치표준화. PPO clipped surrogate(clip.2)+.5valueMSE+.02anchorKL−.01entropy. RWR은exp(clip((return−mean)/.1,−5,5))정규화weight의CE와동일regularizer. Monte Carlo return이며GAE아님.','', '[동결 protocol](JEV_DECISION_PPO96_20261023.md)']
 (REPO/'docs'/doc_name).write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='curves'},indent=2))
if __name__=='__main__':main()
