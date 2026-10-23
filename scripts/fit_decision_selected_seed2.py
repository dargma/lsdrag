"""Independent collection/fit seed for the validation-selected policy; no reselection."""
import json
from pathlib import Path
import torch
from run_decision_policy_pilot import ROOT,save,sha
from fit_decision_state_heads import load_head
from lsdrag.decision_state_learning import StateHead,action_values,expected_utility_loss
from lsdrag.decision_policy_followup import state_weights
from lsdrag.decision_policy import distribution
RUN=ROOT/'results/decision_state_outcomes_seed2_20261023';OUT=ROOT/'results/decision_state_heads_20261023'
def main():
 torch.set_num_threads(2);torch.manual_seed(20261048)
 assert json.loads((RUN/'status.json').read_text())['phase']=='complete'
 m=json.loads((OUT/'manifest.json').read_text());selection=json.loads((OUT/'final_candidate_selection_before_seed2.json').read_text());primary=selection['primary'];spec=m['candidates'][primary];name=primary+'_seed2';assert name not in m['candidates']
 rows=torch.load(RUN/'states.pt',weights_only=True);x=torch.stack([r['pack'] for r in rows]).float();legal=torch.stack([r['legal'] for r in rows]);w=state_weights([r['qid'] for r in rows],'cpu');base=load_head(m['candidates']['grpo_baseline'])
 with torch.no_grad():ref=distribution(base(x,legal)[0],legal).probs.detach()
 if spec['kind']=='structure':
  h=StateHead(inputs=spec['inputs'],body=spec['body']);h.normalize_from_training(x);opt=torch.optim.AdamW(h.parameters(),lr=.001)
  for epoch in range(400):
   opt.zero_grad();p=distribution(h(x,legal)[0],legal).probs;loss=(w*(ref*(ref.clamp_min(1e-9).log()-p.clamp_min(1e-9).log())).sum(-1)).sum();loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.);opt.step()
 else:
  h=base
  for n,p in h.named_parameters():p.requires_grad_(n.startswith(('action.','skip.')))
 warm=OUT/f'{name}_warmstart.pt';torch.save(h.state_dict(),warm)
 lam=spec.get('lambda',.03);kind=spec.get('cost_mode','proxy');q,quality,cost=action_values(rows,lam)
 if kind=='generated':
  for i,r in enumerate(rows):
   for a in range(4):
    vals=[s['generated']/100 for s in r['samples'] if s['action']==a]
    if vals:cost[i,a]=sum(vals)/len(vals)
  q=quality-lam*cost
 opt=torch.optim.AdamW([p for p in h.parameters() if p.requires_grad],lr=1e-4)
 for epoch in range(100):
  opt.zero_grad();loss=expected_utility_loss(h(x,legal)[0],legal,q,ref,w);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.);opt.step()
 cp=OUT/f'{name}.pt';torch.save(h.state_dict(),cp)
 m['candidates'][name]={k:v for k,v in spec.items() if k in ['kind','inputs','body','lambda','cost_mode']}|{'checkpoint':str(cp),'sha256':sha(cp),'training_data_sha256':sha(RUN/'states.pt'),'fit_seed':20261048,'collection_seed':20261047,'selection_fixed_before_seed2':True,'source_sha256':sha(Path(__file__))}
 save(OUT/'manifest.json',m);save(OUT/'seed2_fit.json',{'candidate':name,'primary':primary,'states':len(rows),'data_sha256':sha(RUN/'states.pt'),'fit_seed':20261048});print(name)
if __name__=='__main__':main()
