"""Validate frozen STOP-only results and generate an auditable report."""
import json,hashlib,math
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/content/local_fast/lsdrag-text-v1/results/decision_policy_stop_20261023')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 s=read(ROOT/'summary.json');m=read(ROOT/'manifest.json');status=read(ROOT/'status.json');train=read(ROOT/'training_summary.json')
 assert status['phase']=='complete'
 assert sha(ROOT/'stop_only_round1.pt')==read(ROOT/'frozen_evaluation.json')['stop_sha256']
 hashes={};checked=0;total=0
 for group in ['validation','test']:
  rows=[read(p) for p in (ROOT/group).glob('*.json')];assert len(rows)==len(m[group])*4;total+=len(rows)
  for name in m['policies']:
   eps=[r for r in rows if r['policy']==name];assert {r['qid'] for r in eps}==set(m[group])
   for k in ['f1','em','tool_calls','generated_tokens','prompt_tokens','forced_prefix_tokens','cost_units']:
    assert math.isclose(float(np.mean([r[k] for r in eps])),s['policies'][group][name][k],abs_tol=1e-8)
   for ep in eps:
    for d in ep.get('decisions',[]):
     p=d['policy_probs'];assert abs(sum(p)-1)<1e-5;assert d['legal'][d['action']];assert all(0<=v<=1 for v in p);assert all(v==0 for v,l in zip(p,d['legal']) if not l)
     if name=='stop_only':assert sum(v>0 for v in p[:3])<=1
     checked+=1
  hashes.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/group).glob('*.json')})
 report={**s,'status':status,'training':train,'checks':{'raw_evaluation_episodes':total,'head_probability_checks':checked,'frozen_stop_checkpoint_match':True},'evaluation_sha256':hashes}
 (REPO/'lab/reports/decision_policy_20261023/stop_control.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# 終了 전용 대조군 결과 — 2026-10-23','','[실험 protocol](JEV_DECISION_POLICY_STOP_CONTROL_20261023.md), [PPO·현재 학습·STOPgate 수식](JEV_DECISION_POLICY_MATH_20261023.md), [검증 수치](../lab/reports/decision_policy_20261023/stop_control.json).','','## 새로운 frozen60 결과','','| 조건 | F1 | EM | 호출/문항 | 생성/문항 | input/문항 | 비용proxy | cap | 오류 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,r in s['policies']['test'].items():lines.append(f"| {name} | {r['f1']:.4f} | {r['em']:.3f} | {r['tool_calls']:.3f} | {r['generated_tokens']:.2f} | {r['prompt_tokens']:.1f} | {r['cost_units']:.3f} | {r['forced_final']} | {r['errors']} |")
 lines+=['','## 사전 지정 paired 비교','','| 차이 | F1 delta | 95% bootstrap CI | 승/패 | 호출 delta |','|---|---:|---|---|---:|']
 for name,r in s['comparisons'].items():lines.append(f"| {name} | {r['f1_delta']:+.5f} | [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] | {r['wins']}/{r['losses']} | {r['calls_delta']:+.3f} |")
 lines+=['','F1 신뢰구간이0을 포함하면 품질 차이를 확정하지 않는다. 이 CI는 질문 bootstrap이며 학습seed 변동을 포함하지 않는다. 작게 관측된 품질 손실을 사후 noninferiority로 이름 붙이지 않는다.','','## 검증과 해석 범위','',f"학습은 pilot2 train139상태/990분기의2round 재사용. 초기139상태BC argmax parity와학습후고정도구불변통과. 실행중nonFINAL 도구선택도BC와대조. 새branch수집0이지만원래수집비용이없어진것은아니다. {total}평가원본 평균재집계,{checked}head결정확률검사,checkpoint hash검증통과. worker elapsed{status['elapsed_seconds']:.2f}초(load포함),생성{status['generated']}/input{status['inputs']}.",'','STOP-only는CONT를선택했을때historicalBC의합법nonFINAL argmax를그대로실행한다. 전체MLPcost는pilot2동결checkpoint다. Gate초기화/출력제약이전체정책과달라순수단일요인의인과적ablation이라주장하지않는다. Target은공유reference suffix의Q이며stop-only의on-policyQ가아니다. 단일seed,질문60,참조정책분포및작은학습데이터한계가있다.','', '새60은이제노출평가데이터이며후속선형value/보상/threshold튜닝에쓰지않는다. 호출감소와전체토큰/시간감소를구분하고,cap강제종료는학습된STOP으로세지않는다. 추가모델/새Qwen교체없음. latency는통제측정이아니다.','','재생성: `python scripts/reporting/summarize_decision_policy_stop.py`.']
 math_text=(REPO/'docs/JEV_DECISION_POLICY_MATH_20261023.md').read_text()
 lines += ['', math_text[math_text.index('## 6. 다음'):]]
 (REPO/'docs/JEV_DECISION_POLICY_STOP_REPORT_20261023.md').write_text('\n'.join(lines).replace('終了','종료')+'\n')
 print(json.dumps(report['checks']))
if __name__=='__main__':main()
