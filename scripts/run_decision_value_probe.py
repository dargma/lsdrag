"""Question-family OOF action-value prediction; frozen CPU representation comparison."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];ROOT=Path('/content/local_fast/lsdrag-text-v1');REG=REPO/'lab/registry/decision_value_probe_20261023';OUT=REPO/'lab/reports/decision_value_probe_20261023'
DATA=ROOT/'results/decision_policy_20261023_pilot2/round0/counterfactual_data.pt';HEAD=ROOT/'results/improvement_v3_merged_20261017/head.pt';REPEATS=ROOT/'results/decision_reliability_20261023/samples.json'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def predict(x,y,qids,xt,penalty=1.):
 # Signed Q regression: no non-negative clipping (utility may be negative).
 w=np.array([1/qids.count(q) for q in qids]);w/=w.sum();mu=w@x;sd=np.sqrt(w@((x-mu)**2)).clip(1e-6);ym=w@y
 X=np.sqrt(w)[:,None]*(x-mu)/sd;Y=np.sqrt(w)*(y-ym);coef=X.T@np.linalg.solve(X@X.T+penalty*np.eye(len(x)),Y)
 return (xt-mu)/sd@coef+ym

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
 if a.prepare:
  prior=read(REPO/'lab/registry/decision_acquisition_probe_20261023/manifest.json')
  m={'seed':20261029,'folds':prior['folds'],'families':prior['families'],'methods':['action_mean','metadata','last','question_end','observation_end','last16_mean','observation_mean','context_bin_mean'],'primary':'last vs metadata and historical BC','ridge_penalty':1.,'target':'mean(F1-.03cost) per legal action; no clipping','training_data':'round0 two repeats only; extra six repeats evaluation only','primary_metric':'question-balanced OOF selected-action utility gain over BC','secondary':'OOF evaluation on12 repeat-audited states, all6 fresh suffixes','gate':'primary last gain vs BC and metadata both have question bootstrap lower95%>0; no method selection from auxiliary comparisons','scope':'retrospective train-family-held-out head fit conditional on teacher-trained frozen BC/reference; not pristine unseen end-to-end evaluation','input_sha256':{str(p):sha(p) for p in [DATA,HEAD,REPEATS]},'source_sha256':{str(Path(__file__).relative_to(REPO)):sha(Path(__file__))}}
  assert not (REG/'manifest.json').exists();save(REG/'manifest.json',m);return
 m=read(REG/'manifest.json');assert all(sha(Path(p))==h for p,h in m['input_sha256'].items());assert all(sha(REPO/p)==h for p,h in m['source_sha256'].items());torch.set_num_threads(2)
 rows=torch.load(DATA,weights_only=True,map_location='cpu');h=torch.load(HEAD,weights_only=True,map_location='cpu');n=len(rows);legal=np.stack([r['legal'].numpy() for r in rows]);qids=[r['qid'] for r in rows]
 packs=np.stack([r['pack'].float().numpy() for r in rows]).astype(np.float64);Q=np.full((n,4),np.nan);bc=[];meta=[]
 for i,r in enumerate(rows):
  logits=(h['action_weight']@r['pack'][0].float()+h['action_bias']).numpy();logits[~legal[i]]=-1e9;p=np.exp(logits-logits.max());p/=p.sum();bc.append(int(p.argmax()));meta.append(np.r_[r['turn'],legal[i],p])
  for act in np.flatnonzero(legal[i]):Q[i,act]=np.mean([s['f1']-.03*s['cost'] for s in r['samples'] if s['action']==act])
 meta=np.array(meta);features={'metadata':meta,**{name:packs[:,i] for i,name in enumerate(m['methods'][2:7])},'context_bin_mean':packs[:,5:].mean(1)}
 pred={name:np.full((n,4),np.nan) for name in m['methods']}
 for held in m['folds']:
  te=np.array([i for i,q in enumerate(qids) if q in held]);tr=np.array([i for i,q in enumerate(qids) if q not in held]);assert not {f for i in tr for f in m['families'][qids[i]]}&{f for i in te for f in m['families'][qids[i]]}
  for act in range(4):
   train=tr[legal[tr,act]];test=te[legal[te,act]]
   if not len(test):continue
   assert len(train)>0
   qs=[qids[i] for i in train];weights=np.array([1/qs.count(q) for q in qs]);weights/=weights.sum();pred['action_mean'][test,act]=weights@Q[train,act]
   for name,x in features.items():pred[name][test,act]=predict(x[train],Q[train,act],qs,x[test],m['ridge_penalty'])
 extra=read(REPEATS);detail=[]
 for i,r in enumerate(rows):
  choices={name:int(np.where(legal[i],p[i],-np.inf).argmax()) for name,p in pred.items()};base=Q[i,bc[i]]
  e=[s for s in extra if s['qid']==r['qid'] and s['turn']==r['turn']];eq={act:np.mean([s['utility'] for s in e if s['action']==act]) for act in np.flatnonzero(legal[i])} if e else None
  detail.append({'qid':r['qid'],'turn':r['turn'],'bc':bc[i],'choices':choices,'gains':{k:float(Q[i,c]-base) for k,c in choices.items()},'regrets':{k:float(np.nanmax(Q[i])-Q[i,c]) for k,c in choices.items()},'q_mse':{k:float(np.mean((p[i,legal[i]]-Q[i,legal[i]])**2)) for k,p in pred.items()},'extra_gains':{k:float(eq[c]-eq[bc[i]]) for k,c in choices.items()} if eq else None})
 unique=sorted(set(qids));rng=np.random.default_rng(m['seed']);boot=rng.integers(len(unique),size=(4000,len(unique)));summary={};gains={}
 for name in m['methods']:
  g=np.array([np.mean([r['gains'][name] for r in detail if r['qid']==q]) for q in unique]);gains[name]=g
  eg=[r['extra_gains'][name] for r in detail if r['extra_gains'] is not None]
  summary[name]={'gain_over_bc':float(g.mean()),'ci95':np.quantile(g[boot].mean(1),[.025,.975]).tolist(),'regret':float(np.mean([np.mean([r['regrets'][name] for r in detail if r['qid']==q]) for q in unique])),'q_mse':float(np.mean([np.mean([r['q_mse'][name] for r in detail if r['qid']==q]) for q in unique])),'extra6_gain':float(np.mean(eg)),'extra6_states':len(eg)}
 delta=gains['last']-gains['metadata'];ci=np.quantile(delta[boot].mean(1),[.025,.975]).tolist();passed=summary['last']['ci95'][0]>0 and ci[0]>0
 result={'summary':summary,'last_minus_metadata':{'mean':float(delta.mean()),'ci95':ci},'gate_passed':bool(passed),'states':n,'questions':len(unique),'detail':detail,'scope':m['scope']};save(OUT/'result.json',result)
 lines=['# 질문 분리 행동 가치 예측 결과','',m['scope'],'','동일67상태/24질문/6family fold,동일ridgeρ1·action별 signed utility target. 추가6회 outcome은 fitting에 사용하지 않음.','', '| 표현/기준 | BC대비 utility gain | 95% CI | regret | Q MSE | 추가6회12상태 gain |','|---|---:|---|---:|---:|---:|']
 for name,s in summary.items():lines.append(f"| {name} | {s['gain_over_bc']:.4f} | {s['ci95']} | {s['regret']:.4f} | {s['q_mse']:.4f} | {s['extra6_gain']:.4f} |")
 lines+=['',f'주조건 last−metadata: {delta.mean():.4f}, CI {ci}. Gate passed: {passed}.','', 'CI는고정OOF예측의질문bootstrap이며재학습/독립수집seed변동을포함하지않는다. 사후최고aux표현을확정채택하지않는다. Qtarget은고정reference suffix결과이며,새head가이후모든행동을결정하는closed-loop성능이아니다. 정책선택확률/성공확률calibration도검증하지않았다.','', '[사전 계획](JEV_DECISION_VALUE_PROBE_20261023.md)']
 (REPO/'docs/JEV_DECISION_VALUE_PROBE_REPORT_20261023.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='detail'},indent=2))
if __name__=='__main__':main()
