#!/usr/bin/env python
"""Family-held-out train-pool acquisition probe. CPU only; no agent evaluation."""
import argparse,hashlib,json,math,os
from pathlib import Path
import numpy as np
import torch
from lsdrag.decision_acquisition import family_components,balanced_folds,fit_predict_ridge,exact_budget,uniform_inclusion,select_exact
REPO=Path(__file__).resolve().parents[1];ROOT=Path('/content/local_fast/lsdrag-text-v1');RUN=ROOT/'results/decision_policy_20261023_pilot2'
REG=REPO/'lab/registry/decision_acquisition_probe_20261023';OUT=REPO/'lab/reports/decision_acquisition_probe_20261023'
HEAD=ROOT/'results/improvement_v3_merged_20261017/head.pt'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def prepare():
 m=read(RUN/'manifest.json');qids=m['train'];families={}
 for line in (ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open():
  q=json.loads(line)
  if q['id'] in qids:families[q['id']]=[d['id'] for d in q['question_decomposition']]
 assert set(families)==set(qids)
 rows=torch.load(RUN/'round0/counterfactual_data.pt',weights_only=True,map_location='cpu');counts={q:sum(r['qid']==q for r in rows) for q in qids}
 groups=family_components(qids,families);folds=balanced_folds(groups,counts)
 files=[RUN/f'round{r}/counterfactual_data.pt' for r in range(2)]+[HEAD]
 source=['scripts/run_decision_acquisition_probe.py','src/lsdrag/decision_acquisition.py']
 p={'profile':'decision_acquisition_probe_20261023','seed':20261027,'families':families,'groups':groups,'folds':folds,'train_questions':qids,'primary_round':0,'primary_learner':'hidden_metadata_ridge','ridge_penalty':1.,'target':'mean action Q gap over BC; Q=F1-.03cost','primary_score':'signed cross-repeat selection/evaluation BC gain at25% branch budget','budget_fractions':[.25,.5],'cost':'2 * number of legal actions; exact same attainable branch count per fold','token_equal_budget':False,'methods':['uniform_exact_expectation','BC_entropy','early_turn','metadata_ridge','hidden_metadata_ridge','posthoc_mean_target'],'gate':'at25%, hidden_metadata exceeds uniform and entropy total signed gain; >=4/6 folds beat uniform; family bootstrap lower95% bound vs uniform >0','gate_only_authorizes':'design a prospective bounded collection comparison, not a success claim','round1':'diagnostic only; reference was trained using all train questions outcomes','new_LLM_inference':False,'old_policy_evaluation_data_used':False,'input_sha256':{str(p):sha(p) for p in files},'source_sha256':{str(p):sha(REPO/p) for p in source}}
 if (REG/'manifest.json').exists():raise FileExistsError('Frozen manifest exists')
 save(REG/'manifest.json',p);print(json.dumps({'independent_groups':len(groups),'fold_question_counts':[len(f) for f in folds],'fold_states':[sum(counts[q] for q in f) for f in folds]}))

def records(rid,hist):
 rows=torch.load(RUN/f'round{rid}/counterfactual_data.pt',weights_only=True,map_location='cpu');result=[]
 for r in rows:
  legal=np.flatnonzero(r['legal'].numpy()).tolist();h=r['pack'][0].float();logits=hist['action_weight']@h+hist['action_bias'];logits[~r['legal']]=-1e9;p=logits.softmax(-1).numpy();bc=int(logits.argmax());b=legal.index(bc)
  u=np.array([[s['f1']-.03*s['cost'] for s in r['samples'] if s['action']==a] for a in legal]);assert u.shape==(len(legal),2)
  f=np.array([[s['f1'] for s in r['samples'] if s['action']==a] for a in legal]);c=np.array([[s['cost'] for s in r['samples'] if s['action']==a] for a in legal])
  choices=u.argmax(0);cross=float(np.mean([u[choices[j],1-j]-u[b,1-j] for j in range(2)]));entropy=float(-(p*np.log(p.clip(1e-30))).sum());sortedp=np.sort(p)
  meta=np.r_[r['turn'],len(legal),p,entropy,sortedp[-1]-sortedp[-2]]
  collector=read(RUN/f'round{rid}/collectors/reference_{r["qid"]}.json');history=[{k:v[k] for k in ['tool_name','arguments','result']} for v in collector['trajectory'][:r['turn']]]
  prefix=hashlib.sha256(json.dumps({'qid':r['qid'],'history':history},sort_keys=True).encode()).hexdigest()
  result.append({'qid':r['qid'],'turn':r['turn'],'prefix':prefix,'round':rid,'legal':legal,'cost':len(legal)*2,'h':h.numpy(),'meta':meta,'y':float(u.mean(1).max()-u.mean(1)[b]),'cross':cross,'entropy':entropy,'utility':u.mean(1).tolist(),'quality':f.mean(1).tolist(),'remaining_cost':c.mean(1).tolist(),'best':legal[int(u.mean(1).argmax())]})
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args();torch.set_num_threads(2)
 if a.prepare:prepare();return
 m=read(REG/'manifest.json');assert all(sha(REPO/p)==h for p,h in m['source_sha256'].items());assert all(sha(Path(p))==h for p,h in m['input_sha256'].items())
 hist=torch.load(HEAD,map_location='cpu',weights_only=True);rows=records(0,hist);assert {r['qid'] for r in rows}==set(m['train_questions'])
 oof={name:np.zeros(len(rows)) for name in ['metadata_ridge','hidden_metadata_ridge']};y=np.array([r['y'] for r in rows]);cross=np.array([r['cross'] for r in rows]);fold_records=[]
 for fi,held in enumerate(m['folds']):
  te=np.array([i for i,r in enumerate(rows) if r['qid'] in held]);tr=np.array([i for i,r in enumerate(rows) if r['qid'] not in held]);assert not {rows[i]['qid'] for i in tr}&set(held)
  train_families={f for i in tr for f in m['families'][rows[i]['qid']]};test_families={f for i in te for f in m['families'][rows[i]['qid']]};assert not train_families&test_families
  for name in oof:
   x=np.stack([r['meta'] if name=='metadata_ridge' else np.r_[r['h'],r['meta']] for r in rows]);pred,_=fit_predict_ridge(x[tr],y[tr],[rows[i]['qid'] for i in tr],x[te],penalty=1.);oof[name][te]=pred
  costs=[rows[i]['cost'] for i in te]
  for frac in m['budget_fractions']:
   budget=exact_budget(costs,frac);masks={'uniform_exact_expectation':uniform_inclusion(costs,budget)}
   score={'BC_entropy':np.array([rows[i]['entropy'] for i in te]),'early_turn':np.array([1/(1+rows[i]['turn']) for i in te]),'posthoc_mean_target':y[te],**{name:p[te] for name,p in oof.items()}}
   masks.update({name:select_exact(s,costs,budget) for name,s in score.items()})
   for name,mask in masks.items():
    assert abs(mask@np.array(costs)-budget)<1e-7
    fold_records.append({'fold':fi,'fraction':frac,'method':name,'branch_budget':budget,'pool_branch_cost':sum(costs),'expected_or_actual_selected_states':float(mask.sum()),'signed_cross_gain':float(mask@cross[te]),'observed_mean_target_gain':float(mask@y[te]),'indices':te.tolist(),'mask':mask.tolist()})
 summary={}
 for frac in m['budget_fractions']:
  summary[str(frac)]={}
  for name in m['methods']:
   rs=[r for r in fold_records if r['fraction']==frac and r['method']==name]
   summary[str(frac)][name]={k:sum(r[k] for r in rs) for k in ['branch_budget','pool_branch_cost','expected_or_actual_selected_states','signed_cross_gain','observed_mean_target_gain']}
 contributions={name:np.zeros(len(rows)) for name in m['methods']}
 for r in fold_records:
  if r['fraction']==.25:contributions[r['method']][r['indices']]=np.array(r['mask'])*cross[r['indices']]
 delta=contributions['hidden_metadata_ridge']-contributions['uniform_exact_expectation'];groups=m['groups'];gd=np.array([sum(delta[i] for i,r in enumerate(rows) if r['qid'] in g) for g in groups]);rng=np.random.default_rng(m['seed']);boots=gd[rng.integers(0,len(gd),size=(4000,len(gd)))].sum(-1);ci=np.quantile(boots,[.025,.975]).tolist()
 wins=0
 for f in range(6):
  rr={r['method']:r for r in fold_records if r['fold']==f and r['fraction']==.25}
  wins+=rr['hidden_metadata_ridge']['signed_cross_gain']>rr['uniform_exact_expectation']['signed_cross_gain']
 primary=summary['0.25'];passed=bool(primary['hidden_metadata_ridge']['signed_cross_gain']>max(primary['uniform_exact_expectation']['signed_cross_gain'],primary['BC_entropy']['signed_cross_gain']) and wins>=4 and ci[0]>0)
 # Round1 diagnostics are isolated from acquisition fitting/scoring/gate.
 later=records(1,hist);lookup={r['prefix']:r for r in rows};matched=[(lookup[r['prefix']],r) for r in later if r['prefix'] in lookup]
 drift={'matched_public_prefix_states':len(matched),'best_action_changed':sum(a['best']!=b['best'] for a,b in matched),'mean_abs_action_Q_change':float(np.mean([np.mean(np.abs(np.array(a['utility'])-np.array(b['utility']))) for a,b in matched])),'interpretation':'reference change and two-sample Monte Carlo noise are confounded; not a causal reference effect'}
 serial=[{k:v for k,v in r.items() if k not in ['h','meta']}|{'metadata_features':r['meta'].tolist(),'oof_scores':{n:float(p[i]) for n,p in oof.items()}} for i,r in enumerate(rows)]
 result={'scope':'retrospective train-only family-held-out probe; not prospective rollout savings','states':len(rows),'questions':len(m['train_questions']),'family_groups':len(groups),'summary':summary,'gate':{'passed':passed,'wins_over_uniform_folds':wins,'conditional_family_bootstrap_delta_ci95':ci,'point_delta':float(gd.sum()),'uncertainty_limit':'fixed OOF predictions, overlapping training folds; no refit/seed variance'},'reference_drift_diagnostic':drift,'fold_results':fold_records,'states_detail':serial,'new_agent_inference':False,'old_validation_test_outcomes_used':False}
 save(OUT/'result.json',result)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 names=m['methods'];vals=[primary[n]['signed_cross_gain'] for n in names];fig,ax=plt.subplots(figsize=(8,4));ax.bar(range(len(names)),vals,color=['gray','tab:orange','tab:green','tab:blue','tab:purple','silver']);ax.set_xticks(range(len(names)),['Uniform\nexpectation','BC entropy','Early turn','Metadata\nridge','Hidden+meta\nridge','Post-hoc\ntarget']);ax.set_ylabel('Sum of cross-repeat BC gain on selected states');ax.set_title('Round0 family-held-out probe: equal branch-count budget');ax.axhline(0,color='black',linewidth=.5);fig.tight_layout();OUT.mkdir(parents=True,exist_ok=True);fig.savefig(OUT/'acquisition_comparison.svg');plt.close(fig)
 lines=['# 중요 상태 사전 예측 점검 — 2026-10-23','','**판정: '+('다음 실제 수집 비교를 설계할 gate 통과.' if passed else '사전 지정 gate 미통과. 현재 중요도 예측기로 실제 선택 수집을 확대하지 않는다.')+'** 아직 새 rollout 절약이나 독립 정책 성능을 입증한 실험은 아니다.','','## 주 검증: round0만 사용','','고정 BC reference로 수집한67상태/24질문을 family 연결성분으로6fold 분리. Held-out 질문의 중요도 target은 fitting·정규화·선택에 사용하지 않는다. 단, frozen BC 자체는기존24질문teacher label을학습한모델이므로완전히새질문의end-to-end성능이라고부르지않는다. 이전분기audit로전체자료의집중도는이미관측했으므로이검증도완전한confirmatorytest가아니다.','','각상태를고르면합법행동당2suffix를수집한다고가정. 6/8개의branch수에맞춰모든방법의실제선택비용을정확히같게맞췄다. Token·wallclock비용은아직동등하지않다. Random은같은정확예산을쓰는feasible subset의uniform기댓값이다.','','| 방법 | 분기 예산 | 선택 상태 | Cross-repeat gain 합 | 같은 표본 평균 target gain 합 |','|---|---:|---:|---:|---:|']
 for name,r in primary.items():lines.append(f"| {name} | {r['branch_budget']} | {r['expected_or_actual_selected_states']:.2f} | {r['signed_cross_gain']:.4f} | {r['observed_mean_target_gain']:.4f} |")
 lines+=['','Cross-repeat는한repeat로bestaction을고르고다른repeat로BC대비효용차를평가한값이다. Importance selector는held-out결과를사용하지않지만,최종metric계산에는사후분기결과가필요하다. Post-hoc row는배포가능방법이아니며cross-repeat상한도보장하지않는다.','',f"Hidden+metadata 대uniform의합차이{float(gd.sum()):+.4f},조건부familybootstrap95%CI[{ci[0]:+.4f},{ci[1]:+.4f}],fold승수{wins}/6. CI는고정OOFprediction의family불확실성일뿐재학습·수집seed변동을포함하지않는다.",'','![Acquisition comparison](../lab/reports/decision_acquisition_probe_20261023/acquisition_comparison.svg)','','## 더 넓은 진단','','- Hidden+metadata와작은metadata만의차이는고차원표현이정말필요한지판별한다. 무조건복잡한head로진행하지않는다.','- Round1은전체trainoutcome을보고개선된reference로수집됐으므로주CV에넣지않았다.',f"- 동일공개prefix {len(matched)}개중bestaction변화{drift['best_action_changed']}개,평균actionQ변화{drift['mean_abs_action_Q_change']:.4f}. 이값에는reference변경과MC잡음이섞여있다.",'- 선택상태중요도가학습loss/episode품질개선과같지않다. 실제학습효율의최종판정은같은예산으로모은데이터로policy를학습하고새episode로평가해야한다.','','## 다음 행동','','Gate 미통과면추가selector튜닝으로동일fold를계속소모하지않는다. 반복suffix로target신뢰도를확인하거나종료·Qcoverage문제를별도로진단한다. Gate 통과여도새질문/seed와token예산을고정한실제수집대조가필요하다. 기존결과100/20/40/60/60은튜닝에사용하지않는다.','','[동결 protocol](JEV_DECISION_ACQUISITION_PROBE_20261023.md),[기계판독결과](../lab/reports/decision_acquisition_probe_20261023/result.json).']
 (REPO/'docs/JEV_DECISION_ACQUISITION_PROBE_REPORT_20261023.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'summary':summary,'gate':result['gate'],'drift':drift},indent=2))
if __name__=='__main__':main()
