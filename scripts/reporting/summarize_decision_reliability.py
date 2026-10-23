"""Summarize predeclared split-half reliability, preserving all states."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[2];OUT=ROOT/'results/decision_reliability_20261023'
def read(p):return json.loads(p.read_text())
def main():
 status=read(OUT/'status.json');assert status['phase']=='complete'
 m=read(OUT/'manifest.json');samples=read(OUT/'samples.json');assert len(samples)==m['max_branches']
 old=torch.load(ROOT/'results/decision_policy_20261023_pilot2/round0/counterfactual_data.pt',weights_only=True,map_location='cpu')
 hist=torch.load(ROOT/'results/improvement_v3_merged_20261017/head.pt',weights_only=True,map_location='cpu');result=[]
 for i,s in enumerate(m['states']):
  r=next(r for r in old if r['qid']==s['qid'] and r['turn']==s['turn']);legal=np.flatnonzero(s['legal']);logits=hist['action_weight']@r['pack'][0].float()+hist['action_bias'];logits[~r['legal']]=-1e9;bc=int(logits.argmax());b=list(legal).index(bc)
  values=np.array([[next(x['utility'] for x in samples if x['state']==i and x['action']==a and x['repeat']==rep) for rep in range(6)] for a in legal]);halves=np.stack([values[:,:3].mean(1),values[:,3:].mean(1)],axis=1);choices=halves.argmax(0);cross=float(np.mean([halves[choices[j],1-j]-halves[b,1-j] for j in range(2)]));mean=values.mean(1)
  oldq=np.array([np.mean([v['f1']-.03*v['cost'] for v in r['samples'] if v['action']==a]) for a in legal]);nf=np.flatnonzero(legal!=3);f=list(legal).index(3)
  result.append({'qid':s['qid'],'turn':s['turn'],'bc':bc,'legal':legal.tolist(),'utility_means':mean.tolist(),'utility_sample_sd':values.std(1,ddof=1).tolist(),'cross_half_gain':cross,'optimistic_gain':float(mean.max()-mean[b]),'half_argmax_agreement':bool(choices[0]==choices[1]),'half_selected_within_005_both':bool(all(halves[:,1-j].max()-halves[choices[j],1-j]<=.05 for j in range(2))),'old_new_best_agree':bool(oldq.argmax()==mean.argmax()),'best_nonfinal_minus_final':float(mean[nf].max()-mean[f]),'nonfinal_range':float(np.ptp(mean[nf]))})
 gains=np.array([r['cross_half_gain'] for r in result]);rng=np.random.default_rng(m['seed']);ci=np.quantile(gains[rng.integers(len(gains),size=(4000,len(gains)))].mean(1),[.025,.975]).tolist()
 summary={'states':len(result),'episodes':len(samples),'cross_half_gain_mean':float(gains.mean()),'question_bootstrap_ci95':ci,'optimistic_gain_mean':float(np.mean([r['optimistic_gain'] for r in result])),'half_argmax_agreement':float(np.mean([r['half_argmax_agreement'] for r in result])),'half_selected_within_005_both':float(np.mean([r['half_selected_within_005_both'] for r in result])),'old_new_best_agreement':float(np.mean([r['old_new_best_agree'] for r in result])),'positive_cross_half_states':int((gains>0).sum()),'negative_cross_half_states':int((gains<0).sum()),'status':status,'states_detail':result,'limitation':'12 pre-existing train states, fixed reference; not learned policy gain or independent collector/training-seed replication'}
 target=REPO/'lab/reports/decision_reliability_20261023';target.mkdir(parents=True,exist_ok=True);(target/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 lines=['# 고정 reference 반복 신뢰성 결과','',f'12개 학습 상태, 합법행동당 새6회, 총{len(samples)}회. 정책 학습이나 held-out QA 평가가 아닌 train-only 진단.', '', '| 지표 | 결과 |','|---|---:|']
 for k in ['cross_half_gain_mean','question_bootstrap_ci95','optimistic_gain_mean','half_argmax_agreement','half_selected_within_005_both','old_new_best_agreement','positive_cross_half_states','negative_cross_half_states']:lines.append(f'| {k} | {summary[k]} |')
 lines+=['','Cross-half: 앞3회로 선택한 행동을 뒤3회에서 BC와 비교하고 역방향도 평균. Utility=F1−.03cost. CI는12질문 재표집이며 고정된 reference/상태에 조건부이다.','', '학습 신호가 양수라는 제한적 근거가 있다.' if ci[0]>0 else 'CI가0을 포함하거나 하회하므로 안정적인 양의 학습 신호를 확정하지 않는다.','', '기존2회와 새6회 순위 비교는 보조 진단. Greedy FINAL 반복 일치는 독립 성공확률 증거가 아니다. 정책의 일반화, 독립 수집/학습 seed 재현, PPO/GRPO 우월성을 입증하지 않는다.','', '[사전 protocol](JEV_DECISION_RELIABILITY_20261023.md) · [상태별 수치](../lab/reports/decision_reliability_20261023/result.json)']
 (REPO/'docs/JEV_DECISION_RELIABILITY_REPORT_20261023.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='states_detail'},indent=2))
if __name__=='__main__':main()
