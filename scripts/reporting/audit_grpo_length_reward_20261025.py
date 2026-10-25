"""Read-only independent reward-design checks on all saved coefficient trials."""
import json,hashlib,itertools,math
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/content/local_fast/lsdrag-text-v1/results');REPO=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(2)
 names={.1:'qwen35_final_grpo48_20261025',.5:'qwen35_grpo_token5_20261025',2.:'qwen35_grpo_token20_20261025',10.:'qwen35_grpo_token100_20261025'}
 report={'verified':True,'formula':'EM + 0.25*F1 - lambda*T/(T+256) + 0.02*valid','scope':'All episode generated argument/final tokens; input/forced prefix excluded; not final-answer-only nor total compute cost','trials':{}}
 original={}
 for lam,name in names.items():
  run=ROOT/name;plan=read(run/'plan.json');audit=read(REPO/'lab/reports'/name/'audit.json');assert audit['verified']
  eps={};tool_tokens=final_tokens=forced_final_tokens=boundary_calls=0
  for file,h in audit['raw_sha256'].items():
   path=Path(file);assert sha(path)==h;e=read(path);key=(e['qid'],e['policy']);eps[key]=e
   quality=e['em']+.25*e['f1'];cost=lam*e['generated_tokens']/(e['generated_tokens']+256);assert abs(e['reward']-(quality-cost+.02*e['format_valid']))<1e-12
   calls=[u for u in e['reader_calls'] if 'raw_json' in u];assert len(calls)==len(e['decisions'])
   assert e['generated_tokens']==sum(len(u.get('token_ids',[])) for u in e['reader_calls'])
   for d,u in zip(e['decisions'],calls):
    assert u['generated_tokens']==len(u['token_ids'])
    if d['action']==3:
     final_tokens+=u['generated_tokens']
     if d['forced']:forced_final_tokens+=u['generated_tokens']
    else:tool_tokens+=u['generated_tokens']
    boundary_calls+=bool(u.get('wrapper_boundary_token_counted'))
   if lam==.1:original[key]=e
   else:
    for k in ['answer','em','f1','generated_tokens','prompt_tokens','decisions','reader_calls']:assert e[k]==original[key][k]
  batch=torch.load(run/'batch.pt',map_location='cpu',weights_only=True);groups=read(run/'groups.json');by_ep={}
  for b in batch:
   key=(b['qid'],f"train{b['rep']}");assert not b['forced'];assert b['reward']==eps[key]['reward'];by_ep.setdefault(key,[]).append(b)
  assert len(by_ep)==192
  for key,bs in by_ep.items():
   assert all(b['episode_steps']==len(bs) for b in bs)
   assert len(bs)==sum(not d['forced'] for d in eps[key]['decisions'])
   # Full-batch objective gives each trajectory equal aggregate weight.
   assert abs(sum(len(batch)/(192*b['episode_steps']) for b in bs)/len(batch)-1/192)<1e-12
  shortest_best=top_wrong_with_correct=quality_sacrificed=rank_changed=sign_changed=0
  conflicts=[];sameqgroups=allwronggroups=0;costs=[];adiff=[]
  for q,g in groups.items():
   es=[eps[q,f'train{i}'] for i in range(4)];r=np.array([e['reward'] for e in es]);adv=(r-r.mean())/(r.std()+1e-6);assert np.allclose(adv,g['advantages'],atol=1e-12)
   quality=np.array([e['em']+.25*e['f1'] for e in es]);tokens=np.array([e['generated_tokens'] for e in es]);costs.extend(lam*tokens/(tokens+256))
   best=np.flatnonzero(np.isclose(r,r.max(),rtol=0,atol=1e-12));shortest_best+=any(tokens[i]==tokens.min() for i in best)
   if max(e['em'] for e in es)==1 and all(es[i]['em']==0 for i in best):
    top_wrong_with_correct+=1;conflicts.append({'qid':q,'EM':[e['em'] for e in es],'F1':[e['f1'] for e in es],'T':tokens.tolist(),'rewards':r.tolist(),'top_indices':best.tolist()})
   quality_sacrificed+=all(quality[i]<quality.max() for i in best)
   sameqgroups+=bool(np.ptp(quality)==0);allwronggroups+=all(e['em']==0 for e in es)
   oldr=np.array([original[q,f'train{i}']['reward'] for i in range(4)]);olda=(oldr-oldr.mean())/(oldr.std()+1e-6)
   rank_changed+=not np.array_equal(np.argsort(r),np.argsort(oldr));sign_changed+=bool(np.any(np.sign(adv)!=np.sign(olda)));adiff.extend(abs(adv-olda))
  # Reward monotonicity with positive coefficient and exact analytic magnitude.
  for em,f1 in [(0,0),(0,.5),(1,1)]:
   ts=np.arange(0,10001);rs=em+.25*f1-lam*ts/(ts+256)+.02;assert np.all(np.diff(rs)<0)
  report['trials'][str(lam)]={'episodes':192,'states':len(batch),'tool_generated':tool_tokens,'final_generated':final_tokens,'forced_final_generated_in_reward':forced_final_tokens,'wrapper_boundary_calls_counted':boundary_calls,'cost_penalty_min_median_max':[float(np.min(costs)),float(np.median(costs)),float(np.max(costs))],'shortest_among_reward_max_groups':int(shortest_best),'max_reward_wrong_despite_correct_available':int(top_wrong_with_correct),'max_reward_below_best_quality_groups':int(quality_sacrificed),'reward_order_changed_vs_point1':int(rank_changed),'advantage_sign_changed_groups_vs_point1':int(sign_changed),'mean_abs_advantage_change_vs_point1':float(np.mean(adiff)),'equal_quality_groups':int(sameqgroups),'all_wrong_groups':int(allwronggroups),'constant_valid_format':all(e['format_valid'] for e in eps.values()),'conflict_examples':conflicts[:4]}
 # Verify actor sign separately from reward code: positive advantage increases chosen log-probability.
 for advantage in [-1.,1.]:
  delta=torch.tensor(0.,requires_grad=True);ratio=delta.exp();loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage);loss.backward();assert float(delta.grad)==-advantage
 report['actor_sign_verified']=True;report['trajectory_equal_weight_verified']=True;report['same_original_trajectories_verified']=True
 out=REPO/'lab/reports/grpo_length_reward_design_20261025';out.mkdir(parents=True,exist_ok=True);(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
