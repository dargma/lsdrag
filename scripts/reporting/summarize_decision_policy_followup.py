"""Validate and report frozen pilot2 artifacts; no inference or fitting."""
import json,math,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/content/local_fast/lsdrag-text-v1/results/decision_policy_20261023_pilot2')
REPO=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 s=read(ROOT/'summary.json');m=read(ROOT/'manifest.json');status=read(ROOT/'status.json')
 assert status['phase']=='complete'
 checked=0;hashes={};all_eps=[]
 for group in ['validation','test']:
  eps=[read(p) for p in (ROOT/group).glob('*.json')];all_eps+=eps
  assert len(eps)==len(m[group])*len(m['evaluation_policies'])
  for name in m['evaluation_policies']:
   rows=[e for e in eps if e['policy']==name];assert {e['qid'] for e in rows}==set(m[group])
   for key in ['f1','em','tool_calls','generated_tokens','prompt_tokens','forced_prefix_tokens','cost_units']:
    assert math.isclose(float(np.mean([e[key] for e in rows])),s['policies'][group][name][key],abs_tol=1e-8)
   for ep in rows:
    for d in ep.get('decisions',[]):
     p=d['policy_probs'];assert abs(sum(p)-1)<1e-5
     assert all(0<=x<=1 for x in p)
     assert all(x==0 for x,l in zip(p,d['legal']) if not l)
     assert d['legal'][d['action']]
     checked+=1
  for p in (ROOT/group).glob('*.json'):hashes[str(p.relative_to(ROOT))]=sha(p)
 train=read(ROOT/'collection_training.json')
 assert all(len({v['data_sha256'] for v in r['models'].values()})==1 for r in train)
 for r in train:assert sha(ROOT/f"round{r['round']}/counterfactual_data.pt")==next(iter(r['models'].values()))['data_sha256']
 for p,h in read(ROOT/'frozen_evaluation.json')['head_sha256'].items():assert sha(ROOT/'heads'/f'{p}_round1.pt')==h
 report={**s,'status':status,'validation_checks':{'raw_evaluation_episodes':len(all_eps),'head_decisions_probability_legality':checked,'shared_training_data_hashes_match':True,'frozen_heads_match':True},'collection_training':train,'evaluation_sha256':hashes}
 out=REPO/'lab/reports/decision_policy_20261023/pilot2.json';out.write_text(json.dumps(report,indent=2)+'\n')
 lines=['# Decision policy pilot2 결과 — 2026-10-23','','**비용을 고려한 정책의 평균 품질·비용 동시 개선 신호가 관측됐지만, 통계적으로 확정되지 않았다.** 사전 지정 주조건 MLP cost policy는 BC보다 F1 +0.05856, 도구 호출 −0.5/문항이다. F1 paired95% CI는 [−0.06234,+0.18360]으로0을 포함한다. 단일seed·fresh40 결과이며 일반화/비열등성/논문 novelty 증거로 확대하지 않는다.','','[동결 protocol](JEV_DECISION_POLICY_PILOT2_20261023.md), [검증된 전체 수치](../lab/reports/decision_policy_20261023/pilot2.json). Source fd756eb; run `/content/local_fast/lsdrag-text-v1/results/decision_policy_20261023_pilot2`.','','## Frozen test40 결과','','| 조건 | F1 | EM | 도구/문항 | 생성token/문항 | input/문항 | 비용proxy | cap FINAL | 오류문항 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,r in s['policies']['test'].items():
  lines.append(f"| {name} | {r['f1']:.4f} | {r['em']:.3f} | {r['tool_calls']:.3f} | {r['generated_tokens']:.3f} | {r['prompt_tokens']:.1f} | {r['cost_units']:.3f} | {r['forced_final']} | {r['errors']} |")
 lines+=['','비용proxy는 tool calls + generated/100 + input/10000이다. 토큰 단위별 가중치는 실측 시간·금액이 아니다. supplied prefix 토큰은 별도 기록하며 생성token 절감만으로 전체계산 절감이라고 부르지 않는다. Shared-GPU 시간 관측은 통제 speedup이 아니다.','','## 해석','','- 사전 주조건 `mlp_pg_cost`: BC 대비 호출 약20.6%, 생성token 약5.8%, input 약14.8%, 비용proxy 약17.3% 감소. F1/EM 평균은 상승했으나 정답 안정성은 아직 불확실하다. 오류·cap 종료는0이다.','- `mlp_pg_quality`: BC보다 F1이 높지만 호출도 증가한다. 같은 목적에서 cost항 추가는 평균 호출을 낮추는 대신 F1도 낮췄다. 보상 trade-off가 실제 행동에 반영됐다는 관찰이지 모든 질문에서 Pareto 우세라는 뜻은 아니다.','- outcome CE도 미래 결과 학습이다. MLP CE는 더 높은 F1을 얻었으나 호출5.125~6.175회와 cap 종료22~27/40으로 비용이 컸다. PG가 CE보다 정확도도 우월하다는 결론은 틀리다.','- `linear_pg_cost`는 F1.3725/calls1.9로 유망한 사전 포함 보조조건이다. 이 test를 보고 최적 모델로 선택해 재평가 점수처럼 보고하지 않는다. 복잡한 head가 필요하다는 증거는 없다.','','## 실행·검증과 한계','',f"전체{status['episodes']} episodes: collectors48 + branches990 + val64 + test320. 학습2round, 총{status['elapsed_seconds']:.2f}초(load포함), 생성{status['generated']} / input{status['inputs']} tokens. 두 round의 상태수는67/72. 여섯 learner는 round별 동일 데이터 SHA를 사용했다.",f"원본 평가384개 평균 재집계, head결정{checked}개 확률 합·합법 행동, frozen checkpoint/data hash 검증 통과. final status complete. 재생성: `python scripts/reporting/summarize_decision_policy_followup.py`.",'','본 방법은 shared-reference Monte Carlo return 기반 근사 정책 개선으로, PPO/불편향 on-policy gradient가 아니다. reference는 round0 tempered BC, round1 qualityPG이며 모든 learner에 중립적인 상태분포는 아니다. 두 suffix sample만으로 Q순위가 안정적이라고 보장하지 않는다. Backbone·생성기·분기는 고정; 추가 policy backbone 없음. 행동확률은 calibration된 성공확률이 아니다.','', '원래test100과 pilot1test20에 더해 **이번40도 노출된 평가 데이터**다. 후속 학습·보상/threshold/구조 튜닝에 쓰지 않는다. 후속 후보는 related-work 보고서의 종료전용/선형value/강한최적화Native 대조군이며, 현재 결과에 새 조건을 사후추가하지 않았다.']
 (REPO/'docs/JEV_DECISION_POLICY_PILOT2_REPORT_20261023.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(report['validation_checks']))
if __name__=='__main__':main()
