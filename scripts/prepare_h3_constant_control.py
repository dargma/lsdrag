"""Freeze token-only scalar calibration for H3's conditional stopping residual."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
import torch
from run_decision_policy_pilot import ROOT,REPO,sha,save
from evaluate_decision_stop_value_h2 import load_head
from lsdrag.decision_stop_value import stop_margin

NAME='decision_h3_constant_control_20261024'
def main():
 torch.set_num_threads(2)
 reg=REPO/'lab/registry'/NAME;out=ROOT/'results'/NAME
 reg.mkdir(exist_ok=False);out.mkdir(exist_ok=False)
 h3spec=json.loads((ROOT/'results/decision_budget_stop_h3_20261024/manifest.json').read_text())['candidate'];assert sha(Path(h3spec['checkpoint']))==h3spec['sha256']
 head=load_head(h3spec);records=torch.load(ROOT/'results/decision_budget_stop_h3_inputs_20261024/states.pt',weights_only=True)
 # Only generation and H3/H1 decisions determine calibration proposals; F1 never read.
 records=[r for r in records if not any(s['errors'] for s in r['samples'])]
 x=torch.stack([r['pack'] for r in records]);legal=torch.stack([r['legal'] for r in records]);qids=[r['qid'] for r in records]
 with torch.no_grad():
  margin=stop_margin(head.base,x,legal).numpy();h3choice=stop_margin(head,x,legal).numpy()>0
 weights=np.array([1/len(set(qids))/qids.count(q) for q in qids]);cost=np.array([[next(s['generated'] for s in r['samples'] if s['action']==r['continue_action']),next(s['generated'] for s in r['samples'] if s['action']==3)] for r in records])
 thresholds=np.sort(np.unique(-margin));biases=np.concatenate(([thresholds[0]-1],(thresholds[:-1]+thresholds[1:])/2,[thresholds[-1]+1]))
 target=float(weights@cost[np.arange(len(cost)),h3choice.astype(int)])
 proxies=np.array([weights@cost[np.arange(len(cost)),(margin+b>0).astype(int)] for b in biases])
 chosen=[]
 for ratio in (1.,.85,1.15):
  index=min(range(len(biases)),key=lambda i:(abs(proxies[i]-ratio*target),abs(biases[i]),float(biases[i])))
  if index not in chosen:chosen.append(index)
 policies={'h3_reference':h3spec};descriptors=[]
 for j,i in enumerate(chosen):
  c=copy.deepcopy(head)
  with torch.no_grad():c.stop.weight.zero_();c.stop.bias.fill_(float(biases[i]))
  assert all(torch.equal(v,head.base.state_dict()[k]) for k,v in c.base.state_dict().items())
  with torch.no_grad():
   assert torch.equal(c(x,legal)[0][:,:3],head(x,legal)[0][:,:3])
   z=c.base.net[1](c.base.net[0]((c.base.features(x)-c.base.mean)/c.base.scale))
   assert torch.allclose(c.stop(z),torch.full((len(records),1),float(biases[i])))
  path=out/f'constant_{j}.pt';torch.save(c.state_dict(),path)
  policies[f'constant_{j}']=dict(kind='stop_value_h2',checkpoint=str(path),sha256=sha(path))
  descriptors.append(dict(name=f'constant_{j}',bias=float(biases[i]),proxy_generated=float(proxies[i])))
 parent=json.loads((REPO/'lab/registry/decision_stop_value_h2_collect48_20261024/manifest.json').read_text());train=sorted(parent['train'],key=lambda q:hashlib.sha256(f'20261071:{q}'.encode()).hexdigest())[:24]
 assert not set(train)&set(parent['validation']+parent['test'])
 spec=json.loads((REPO/'lab/registry/decision_budget_stop_h3_20261024/validation_spec.json').read_text());spec.update(run_name='decision_h3_constant_calibration24_20261024',split='train_calibration',qids=train,policies=policies,max_seconds=1800,selection='Token-only calibration of constant FINAL residual; never select using F1')
 spec['source_sha256']={p:sha(REPO/p) for p in spec['source_sha256']}
 save(reg/'calibration_spec.json',spec)
 plan=dict(seed=20261071,calibration_questions=train,training_questions=parent['train'],validation=parent['validation'],test=parent['test'],candidates=descriptors,proxy_h3_generated=target,selection='Smallest absolute train mean-generated difference vsH3; tie smallest absolute bias then name; no F1',cost_match_tolerance=.10,max_candidates=3,dev_retuning=False,base_state_dependent=True,input_sha256={str(ROOT/'results/decision_budget_stop_h3_inputs_20261024/states.pt'):sha(ROOT/'results/decision_budget_stop_h3_inputs_20261024/states.pt'),h3spec['checkpoint']:h3spec['sha256']},source_sha256={'scripts/prepare_h3_constant_control.py':sha(REPO/'scripts/prepare_h3_constant_control.py')})
 save(reg/'plan.json',plan);save(out/'plan.json',plan);save(out/'status.json',dict(phase='complete',note='Only calibration checkpoint construction; no QA or policy claim'))
 print(json.dumps(dict(proxy_target=target,candidates=descriptors,calibration_episodes=24*len(policies)),indent=2))
if __name__=='__main__':main()
