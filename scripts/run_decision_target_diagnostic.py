"""Exploratory nested learning curves: absolute Q versus BC-relative advantage."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from run_decision_value_probe import ROOT,REPO,DATA,HEAD,predict,read,sha,save
REG=REPO/'lab/registry/decision_target_diagnostic_20261023';OUT=REPO/'lab/reports/decision_target_diagnostic_20261023'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
 if a.prepare:
  prior=read(REPO/'lab/registry/decision_value_probe_20261023/manifest.json')
  m={'folds':prior['folds'],'families':prior['families'],'seeds':[20261030,20261031,20261032],'fractions':[.25,.5,1.],'features':['last','metadata'],'targets':['absolute_q','bc_advantage'],'ridge_penalty':1.,'scope':'exploratory repeated-use train-family OOF diagnostic; no confirmatory method selection or PPO efficacy claim','inputs':{str(p):sha(p) for p in [DATA,HEAD]},'source':{str(Path(__file__).relative_to(REPO)):sha(Path(__file__)),'scripts/run_decision_value_probe.py':sha(REPO/'scripts/run_decision_value_probe.py')}}
  assert not (REG/'manifest.json').exists();save(REG/'manifest.json',m);return
 m=read(REG/'manifest.json');assert all(sha(Path(p))==h for p,h in m['inputs'].items());assert all(sha(REPO/p)==h for p,h in m['source'].items());torch.set_num_threads(2)
 rows=torch.load(DATA,weights_only=True,map_location='cpu');hist=torch.load(HEAD,weights_only=True,map_location='cpu');qs=[r['qid'] for r in rows];legal=np.stack([r['legal'].numpy() for r in rows]);n=len(rows);Q=np.full((n,4),np.nan);bc=[];meta=[]
 for i,r in enumerate(rows):
  l=(hist['action_weight']@r['pack'][0].float()+hist['action_bias']).numpy();l[~legal[i]]=-1e9;p=np.exp(l-l.max());p/=p.sum();bc.append(int(p.argmax()));meta.append(np.r_[r['turn'],legal[i],p])
  for act in np.flatnonzero(legal[i]):Q[i,act]=np.mean([s['f1']-.03*s['cost'] for s in r['samples'] if s['action']==act])
 base=Q[np.arange(n),bc];xs={'last':np.stack([r['pack'][0].float().numpy() for r in rows]).astype(float),'metadata':np.array(meta)};ys={'absolute_q':Q,'bc_advantage':Q-base[:,None]};results=[]
 for seed in m['seeds']:
  for fraction in m['fractions']:
   for feature,x in xs.items():
    for target,y in ys.items():
     prediction=np.full((n,4),np.nan);training_errors=[];train_counts=[]
     for held in m['folds']:
      available=sorted(set(qs)-set(held),key=lambda q:hashlib.sha256(f'{seed}:{q}'.encode()).hexdigest());chosen=available[:max(1,int(len(available)*fraction))];tr=np.array([i for i,q in enumerate(qs) if q in chosen]);te=np.array([i for i,q in enumerate(qs) if q in held]);train_counts.append(len(chosen))
      assert not {f for q in chosen for f in m['families'][q]}&{f for q in held for f in m['families'][q]}
      for act in range(4):
       ti=tr[legal[tr,act]];ei=te[legal[te,act]]
       if not len(ei):continue
       assert len(ti)>0,'No action coverage in predetermined subset'
       tq=[qs[i] for i in ti];prediction[ei,act]=predict(x[ti],y[ti,act],tq,x[ei]);fitted=predict(x[ti],y[ti,act],tq,x[ti]);training_errors.extend((fitted-y[ti,act])**2)
     assert np.isfinite(prediction[legal]).all();choice=np.where(legal,prediction,-np.inf).argmax(1);gain=Q[np.arange(n),choice]-base;unique=sorted(set(qs));g=np.array([gain[np.array(qs)==q].mean() for q in unique]);mse=np.array([np.mean((prediction[i,legal[i]]-y[i,legal[i]])**2) for i in range(n)])
     results.append({'seed':seed,'fraction':fraction,'feature':feature,'target':target,'training_questions_per_fold':train_counts,'gain':float(g.mean()),'question_gains':g.tolist(),'train_mse_diagnostic_unweighted':float(np.mean(training_errors)),'heldout_mse':float(np.mean([mse[np.array(qs)==q].mean() for q in unique]))})
 summary=[]
 for fraction in m['fractions']:
  for feature in xs:
   for target in ys:
    rr=[r for r in results if r['fraction']==fraction and r['feature']==feature and r['target']==target];summary.append({'fraction':fraction,'feature':feature,'target':target,'gain_mean':float(np.mean([r['gain'] for r in rr])),'gain_seed_min':min(r['gain'] for r in rr),'gain_seed_max':max(r['gain'] for r in rr),'train_mse':float(np.mean([r['train_mse_diagnostic_unweighted'] for r in rr])),'heldout_mse':float(np.mean([r['heldout_mse'] for r in rr]))})
 # Full data must recover previous absolute-Q probe, independently of subset seed.
 prior=read(REPO/'lab/reports/decision_value_probe_20261023/result.json')
 for r in results:
  if r['fraction']==1 and r['target']=='absolute_q':assert np.isclose(r['gain'],prior['summary'][r['feature']]['gain_over_bc'])
 save(OUT/'result.json',{'summary':summary,'runs':results,'scope':m['scope'],'full_data_previous_parity':True})
 lines=['# 학습 데이터 양·상대 이득 target 진단','',m['scope'],'','동일ridgeρ1,학습질문5/10/20개(fold별),3개subset순서. 평가질문은고정. 100%에서3seed는같은자료의중복이며독립학습seed증거가아니다.','', '| fraction | feature | target | gain mean | subset seed range | train MSE | heldout MSE |','|---|---|---|---:|---|---:|---:|']
 for r in summary:lines.append(f"| {r['fraction']} | {r['feature']} | {r['target']} | {r['gain_mean']:.4f} | [{r['gain_seed_min']:.4f}, {r['gain_seed_max']:.4f}] | {r['train_mse']:.4f} | {r['heldout_mse']:.4f} |")
 lines+=['','상대target=Q(s,a)−Q(s,BC(s)). 같은 상태의 행동 순위는 유지하면서 상태별 공통 성분을 제거한다.','', '상대target은 같은 상태의BC가치를 빼며 정답/후속결과는target으로만사용한다. PPO advantage estimator 자체나 PPO학습의재현이아니다. target별분산이다르므로MSE절대크기를서로비교해우월성을판정하지않는다. train MSE는row-level진단,heldout MSE는질문균등. 작은nestedcurve로더많은데이터가반드시해결한다고추정하지않는다. 모든결과는이미여러번본trainpool의탐색적진단.','', '[계획](JEV_DECISION_TARGET_DIAGNOSTIC_20261023.md)']
 (REPO/'docs/JEV_DECISION_TARGET_DIAGNOSTIC_REPORT_20261023.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
