"""Train-only descriptive branching audit. Never reads validation/test outcomes."""
import json,hashlib,math
from pathlib import Path
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/content/local_fast/lsdrag-text-v1')
RUN=ROOT/'results/decision_policy_20261023_pilot2'
OUT=REPO/'lab/reports/decision_policy_20261023'
ACTIONS=['keyword_search','semantic_search','read_chunk','FINAL']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(2)
 m=json.loads((RUN/'manifest.json').read_text());headpath=ROOT/'results/improvement_v3_merged_20261017/head.pt'
 hist=torch.load(headpath,map_location='cpu',weights_only=True)
 states=[];inputs={}
 for rid in range(2):
  path=RUN/f'round{rid}/counterfactual_data.pt';inputs[str(path)]=sha(path)
  rows=torch.load(path,weights_only=True,map_location='cpu')
  for r in rows:
   assert r['qid'] in m['train'] and r['qid'] not in m['validation']+m['test']
   legal=[i for i,v in enumerate(r['legal']) if v]
   f=np.array([[s['f1'] for s in r['samples'] if s['action']==a] for a in legal]);c=np.array([[s['cost'] for s in r['samples'] if s['action']==a] for a in legal]);assert f.shape==(len(legal),2)
   u=f-.03*c;mean=u.mean(1);quality=f.mean(1)
   logits=hist['action_weight']@r['pack'][0].float()+hist['action_bias'];logits[~r['legal']]=-1e9
   bc=int(logits.argmax());bi=legal.index(bc);p=logits.softmax(-1).numpy();best=int(mean.argmax());alt=[j for j in range(len(legal)) if j!=bi]
   selected=[int(u[:,rep].argmax()) for rep in range(2)]
   cross=[float(u[selected[rep],1-rep]-u[bi,1-rep]) for rep in range(2)]
   both=[j for j in alt if bool(np.all(u[j]-u[bi]>.05))]
   nonfinal=[j for j,a in enumerate(legal) if a!=3]
   sign_pairs=[bool((u[j,0]-u[bi,0])*(u[j,1]-u[bi,1])>0) for j in alt if abs(u[j,0]-u[bi,0])>1e-8 and abs(u[j,1]-u[bi,1])>1e-8]
   states.append({'round':rid,'qid':r['qid'],'turn':r['turn'],'legal':legal,'bc_action':bc,'best_mean_action':legal[best],
    'quality_range':float(np.ptp(quality)),'utility_range':float(np.ptp(mean)),'estimated_BC_regret':float(mean.max()-mean[bi]),
    'winner_gap':float(np.sort(mean)[-1]-np.sort(mean)[-2]),'cross_repeat_gain':float(np.mean(cross)),
    'cross_repeat_each':cross,'strict_argmax_agreement':selected[0]==selected[1],
    'epsilon_optimal_set_overlap':bool(set(np.flatnonzero(u[:,0]>=u[:,0].max()-1e-8))&set(np.flatnonzero(u[:,1]>=u[:,1].max()-1e-8))),
    'consistent_alternative_gain_gt05':bool(both),'positive_gap_alt_signs':sign_pairs,
    'BC_entropy':float(-(p*np.log(np.maximum(p,1e-30))).sum()),
    'nonfinal_quality_range':float(np.ptp(quality[nonfinal])) if len(nonfinal)>1 else 0.,
    'Q_quality':dict(zip([ACTIONS[a] for a in legal],quality.tolist())),
    'Q_cost':dict(zip([ACTIONS[a] for a in legal],c.mean(1).tolist())),
    'samples_utility':dict(zip([ACTIONS[a] for a in legal],u.tolist()))})
 qids=sorted({s['qid'] for s in states});counts={q:sum(s['qid']==q for s in states) for q in qids};w=np.array([1/counts[s['qid']]/len(qids) for s in states])
 def summarize(rows):
  regret=np.array([s['estimated_BC_regret'] for s in rows]);rank=np.argsort(-regret)
  return {'states':len(rows),'quality_range_gt0':sum(s['quality_range']>1e-8 for s in rows),'quality_range_ge025':sum(s['quality_range']>=.25 for s in rows),
   'nonfinal_quality_range_ge025':sum(s['nonfinal_quality_range']>=.25 for s in rows),
   'estimated_regret_gt005':int((regret>.05).sum()),'consistent_alternative_gain_gt005':sum(s['consistent_alternative_gain_gt05'] for s in rows),
   'argmax_agreement':float(np.mean([s['strict_argmax_agreement'] for s in rows])),
   'epsilon_optimal_set_overlap':float(np.mean([s['epsilon_optimal_set_overlap'] for s in rows])),
   'estimated_regret_mean':float(regret.mean()),'cross_repeat_gain_mean':float(np.mean([s['cross_repeat_gain'] for s in rows])),
   'top_quarter_share_estimated_regret':float(regret[rank[:math.ceil(len(rows)/4)]].sum()/regret.sum()) if regret.sum() else None,
   'top_quarter_share_note':'post-hoc outcome-ranked descriptive concentration; not prospective selection performance'}
 overall=summarize(states);overall['question_balanced_estimated_regret']=float(w@np.array([s['estimated_BC_regret'] for s in states]));overall['question_balanced_cross_repeat_gain']=float(w@np.array([s['cross_repeat_gain'] for s in states]))
 byround={str(r):summarize([s for s in states if s['round']==r]) for r in range(2)}
 correlations={}
 for feature in ['BC_entropy','turn']:
  x=np.array([s[feature] for s in states]);y=np.array([s['estimated_BC_regret'] for s in states]);correlations[feature]=float(np.corrcoef(x,y)[0,1]) if x.std()>0 else None
 result={'scope':'TRAIN ONLY, no new rollout, descriptive analysis not a learned selector','input_sha256':inputs,'historical_head_sha256':sha(headpath),'n_questions':len(qids),'cost_weight':.03,'suffix_repeats':2,'summary':overall,'by_round':byround,'descriptive_correlations_with_estimated_regret':correlations,'limitations':['two suffix samples are insufficient for stable uncertainty estimates','shared question and reference mean states are not independent','posthoc outcome ranking cannot establish prospective acquisition or policy gain','cross-repeat selection/evaluation is a noise diagnostic, not independent test generalization','cost-reference target differs from deployed learner return'],'states':states}
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 regret=np.sort(np.array([v['estimated_BC_regret'] for v in states]))[::-1]
 xx=np.arange(len(regret)+1)/len(regret);yy=np.r_[0,np.cumsum(regret)]/regret.sum()
 fig,ax=plt.subplots(figsize=(6.7,4.4));ax.plot(xx*100,yy*100,label='Post-hoc outcome ranking',linewidth=2)
 ax.plot([0,100],[0,100],'--',color='gray',label='Expected random selection share')
 k=math.ceil(len(regret)/4);ax.scatter([xx[k]*100],[yy[k]*100],color='tab:orange');ax.annotate(f'{k}/139 states: {yy[k]*100:.1f}% of estimated gap',(xx[k]*100,yy[k]*100),xytext=(30,68),arrowprops={'arrowstyle':'->'})
 ax.set(xlabel='States selected (%)',ylabel='Share of estimated BC improvement gap (%)',title='Train-only concentration; not prospective acquisition performance',xlim=(0,100),ylim=(0,103));ax.legend(loc='lower right');ax.grid(alpha=.2);fig.tight_layout()
 fig.savefig(OUT/'train_branch_value_concentration.svg');plt.close(fig)
 (OUT/'train_branch_value_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 lines=['# 학습 분기 부산물 분석: 중요한 결정이 모여 있는가?','','이 분석은 기존 train24의 두 round139상태/990분기만 읽는다. 평가 결과·정답은 읽지 않았고 새 모델 실행/정책 학습도 없다. 정책학습 성능·novelty 증명이 아니라 후속 연구 가설 점검이다.','','## 정의와 집계','','각 상태의 행동별 평균 utility를 `Q(a)=mean(F1−.03·remaining cost)`로 두고, `max Q−Q(BC 선택)`을 추정 개선 여지로 본다. 행동간 F1범위는 비용과 분리한다. 두 suffix의 한쪽으로 best action을 고른 뒤 다른쪽에서 BC대비 이득을 재계산하고 방향을 뒤집어 평균하여 선택 낙관성을 점검한다. 두 표본만으로 불확실성이 검증된 것은 아니다.','','| 지표 | 결과 |','|---|---:|']
 for k,v in overall.items():
  if k!='top_quarter_share_note':lines.append(f'| {k} | {v:.4f} |' if isinstance(v,float) else f'| {k} | {v} |')
 lines+=['','![Train-only concentration](../lab/reports/decision_policy_20261023/train_branch_value_concentration.svg)','','상위25% 집중도는 **결과를 본 뒤 순위를 정한 분석상 상한**이다. 온라인에서 이 상태들을 미리 고를 수 있다거나 그만큼 rollout을 줄일 수 있다는 뜻이 아니다. 초기BC entropy/turn과의 상관도 동일train상의 기술통계이며 일반화 예측력으로 해석하지 않는다.','','## 다음에 판별해야 할 것','','1. Repeat를 더 수집한 일부 train상태에서 행동 순위와 contrast가 유지되는지 확인한다. 큰 observed gap만 재실행하면 selection bias가 있으므로 작은 gap/무작위 상태도 포함한다.','2. 질문·family 단위 분할로 앞서 얻은 정보만 보는 acquisition score를 학습한다. 평가용 branch outcomes는 수집 선택 전에 숨긴다.','3. 같은 실제 rollout/token 예산의 uniform state 선택과 비교한다. 중요한 상태 발견률뿐 아니라 그 데이터로 학습한 policy의 독립 episode 품질–비용을 측정한다.','4. 독립 reference/수집seed에서도 효과가 남는지 검증한다. 현재139상태와 두샘플만으로 중요한상태의보편적희소성을주장하지않는다.','','[원수치와state별추적](../lab/reports/decision_policy_20261023/train_branch_value_audit.json), [반사실학습수식](JEV_DECISION_POLICY_MATH_20261023.md).']
 (REPO/'docs/JEV_DECISION_BRANCH_VALUE_AUDIT_20261023.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'summary':overall,'by_round':byround,'correlations':correlations},indent=2))
if __name__=='__main__':main()
