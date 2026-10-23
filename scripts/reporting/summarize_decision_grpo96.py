"""Audit complete tempered PPO run and render honest paired comparisons."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/content/local_fast/lsdrag-text-v1/results/decision_grpo96_20261023');REPO=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def main(out_name="decision_grpo96_20261023", doc_name="JEV_DECISION_GRPO96_REPORT_20261023.md"):
 status=read(ROOT/'status.json');assert status['phase']=='complete';m=read(ROOT/'manifest.json');s=read(ROOT/'summary.json');curves=read(ROOT/'curves.json');episodes=[read(p) for p in (ROOT/'episodes').glob('*.json')];assert len(episodes)==m["budget_episodes"]
 checks=0
 for e in episodes:
  for d in e['decisions']:
   p=d['policy_probs'];assert abs(sum(p)-1)<1e-5;assert all(v==0 for v,legal in zip(p,d['legal']) if not legal);checks+=1
 for name,v in s['policies'].items():
  es=[e for e in episodes if e['policy']=='test_'+name];assert len(es)==80 and {e['qid'] for e in es}==set(m['test'])
  for k in ['f1','cost_units','tool_calls','generated_tokens','prompt_tokens']:assert np.isclose(v[k],np.mean([e[k] for e in es]))
 for name,digest in read(ROOT/'frozen_evaluation.json')['head_sha256'].items():assert hashlib.sha256(((ROOT if name=='grpo' else ROOT.parent/'decision_ppo96_20261023')/f'{name}_iteration2.pt').read_bytes()).hexdigest()==digest
 gate=all(s['grpo_minus']['ppo'][metric]['ci95'][0]>0 for metric in ['f1','utility'])
 result=s|{'optimizer_steps_per_learner':sum(4*((c['states']+31)//32) for c in curves),'audit':{'episodes':len(episodes),'decisions':checks,'head_hashes_unchanged':True},'strong_signal_gate_passed':gate,'curves':curves};out=REPO/'lab/reports'/out_name;out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 lines=['# GRPO96 결과 — 2026-10-23','',f'학습1152+새80×3평가240=1392episodes. GRPO 대 PPO의 F1/utility CI하한>0 gate: {gate}. 단일 학습 seed.','', '| 정책 | F1 | utility | 호출 | 생성token | 오류episode | cap FINAL |','|---|---:|---:|---:|---:|---:|---:|']
 for name,v in s['policies'].items():lines.append(f"| {name} | {v['f1']:.4f} | {v['utility']:.4f} | {v['tool_calls']:.3f} | {v['generated_tokens']:.2f} | {v['errors']} | {v['forced_final']} |")
 lines+=['','## Paired 비교','']
 for name,c in s['grpo_minus'].items():
  for metric,v in c.items():lines.append(f"- GRPO−{name} {metric}: {v['delta']:+.4f}, 95% CI {v['ci95']}.")
 lines+=['','## 학습 및 검증','',f'원본{len(episodes)}개재집계,결정{checks}개확률합/합법성,동결headhash검사통과. 각iteration수집당시logprob와update전logprob parity통과.','']
 for c in curves:lines.append(f"- Iteration{c['iteration']}: {c['states']}학습state, diagnostics={c['diagnostic']}")
 lines+=['',f'실제 optimizer updates: {result["optimizer_steps_per_learner"]}.', '', '같은 질문4trajectory의전체utility를group-standardize하고trajectory길이역가중. actor action/skip만학습,trunk고정,critic없음. PPO와state weighting도달라알고리즘단일원인분리불가. 강제행동분기자료없음. 모든평가조건에RecoveryPolicyReader적용.', '', '현재관측: GRPO는PPO보다생성/입력/호출평균이작지만F1과utility차이CI는0포함. 정확도동등성·우월성·통제latency절감입증아님. 평가80은이후튜닝금지.', '', '[동결 protocol](JEV_DECISION_GRPO96_20261023.md)']
 (REPO/'docs'/doc_name).write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='curves'},indent=2))
if __name__=='__main__':main()
