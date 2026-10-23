"""Audit completed value baseline; no fitting or inference."""
import json,hashlib,math
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/content/local_fast/lsdrag-text-v1/results/decision_policy_value_20261023')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 s=read(ROOT/'summary.json');m=read(ROOT/'manifest.json');status=read(ROOT/'status.json');train=read(ROOT/'training_summary.json');assert status['phase']=='complete'
 assert sha(ROOT/'linear_value.pt')==read(ROOT/'frozen_evaluation.json')['value_sha256']
 hashes={};checked=0;total=0
 for group in ['validation','test']:
  rows=[read(p) for p in (ROOT/group).glob('*.json')];assert len(rows)==len(m[group])*5;total+=len(rows)
  for name in m['policies']:
   eps=[r for r in rows if r['policy']==name];assert {r['qid'] for r in eps}==set(m[group])
   for k in ['f1','em','tool_calls','generated_tokens','prompt_tokens','forced_prefix_tokens','cost_units']:
    assert math.isclose(float(np.mean([r[k] for r in eps])),s['policies'][group][name][k],abs_tol=1e-8)
   for ep in eps:
    for d in ep.get('decisions',[]):
     p=d['policy_probs'];assert abs(sum(p)-1)<1e-5;assert d['legal'][d['action']];assert all(0<=v<=1 for v in p);assert all(v==0 for v,l in zip(p,d['legal']) if not l);checked+=1
  hashes.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/group).glob('*.json')})
 report={**s,'status':status,'training':train,'checks':{'raw_evaluation_episodes':total,'head_probability_checks':checked,'frozen_value_checkpoint_match':True},'evaluation_sha256':hashes}
 (REPO/'lab/reports/decision_policy_20261023/value_control.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# 선형 value 대조군 결과 — 2026-10-23','','**선형 value는 정확도가 높았지만 효율적 정책 대체재로 채택할 결과는 아니다.** 새60에서 F1.4426,호출5.20,생성157.2,cap27/60,오류5/60. 전체MLPcost는F1.4059,호출2.10,생성59.02,cap/error0. Value의 BC대비 F1차이CI는양수지만 비용이늘었다. 전체정책대비 F1우위는확정되지않았다.','','[Protocol](JEV_DECISION_POLICY_VALUE_CONTROL_20261023.md), [전체 수치](../lab/reports/decision_policy_20261023/value_control.json), [PPO·정책수식](JEV_DECISION_POLICY_MATH_20261023.md).','','## 새로운 frozen60 결과','','| 조건 | F1 | EM | 호출 | 생성token | input | 비용proxy | cap | 오류 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,r in s['policies']['test'].items():lines.append(f"| {name} | {r['f1']:.4f} | {r['em']:.3f} | {r['tool_calls']:.3f} | {r['generated_tokens']:.2f} | {r['prompt_tokens']:.1f} | {r['cost_units']:.3f} | {r['forced_final']} | {r['errors']} |")
 lines+=['','## 사전 지정 paired 비교','','| 차이 | F1 delta | 95% bootstrap CI | 승/패 | 호출 delta |','|---|---:|---|---|---:|']
 for name,r in s['comparisons'].items():lines.append(f"| {name} | {r['f1_delta']:+.5f} | [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] | {r['wins']}/{r['losses']} | {r['calls_delta']:+.3f} |")
 lines+=['','## 배운 점과 한계','','단순 회귀의 낮은 train MSE가 종료를 포함한 좋은 전체 trajectory를 보장하지 않는다. 종료불안정·reference Q와 배포정책의 차이·희소상태 coverage는 원인 후보이며, 이번 결과만으로 단일원인을 확정하지 않는다. Ridge나value학습 일반이불가능하다는 결론도아니다.','', 'STOP-only는이번별도60에서F1.2680/calls2.1로BC보다품질이낮았다. 첫STOP실험의평균상승이모든질문군에서재현된것은아니다. 반면fullMLPcost의BC대비평균방향은긍정적이나동일한학습seed/checkpoint의다른문항평가이지독립학습재현은아니다.','',f"원본{total}평가평균재집계,head결정{checked}개확률/합법성,checkpoint해시검증통과. Elapsed{status['elapsed_seconds']:.2f}초(load포함),생성{status['generated']},input{status['inputs']}. 시간이통제된speedup결과는아니다.",'','학습은같은139상태의두reference target을pooled회귀했다. 다른정책의순차gradient학습과초기화/estimator가달라방법단독인과비교가아니다. 단일수집seed자료이며닫힌형식해를seed만바꿔반복해도독립재현이되지않는다. Actionsoftmax는정답확률이아니다. 새60은후속튜닝에서제외한다.']
 math_text=(REPO/'docs/JEV_DECISION_POLICY_VALUE_CONTROL_20261023.md').read_text();lines+=['',math_text[math_text.index('## 학습 수식'):math_text.index('[Frozen manifest]')]]
 lines+=['','재생성: `python scripts/reporting/summarize_decision_policy_value.py`.']
 (REPO/'docs/JEV_DECISION_POLICY_VALUE_REPORT_20261023.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report['checks']))
if __name__=='__main__':main()
