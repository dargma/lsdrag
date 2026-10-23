"""Train inexpensive structural candidates on shared frozen all-action outcomes."""
import argparse,copy,json,time
from pathlib import Path
import torch
from lsdrag.decision_policy import PolicyHead,distribution
from lsdrag.decision_state_learning import StateHead,action_values,expected_utility_loss
from lsdrag.decision_policy_followup import state_weights
from run_decision_policy_pilot import ROOT,REPO,save,sha
RUN=ROOT/'results/decision_state_outcomes_20261023';OUT=ROOT/'results/decision_state_heads_20261023'
def read(p):return json.loads(p.read_text())
def load_head(spec,device='cpu'):
 h=PolicyHead() if spec['kind']=='original' else StateHead(inputs=spec['inputs'],body=spec['body'])
 h.load_state_dict(torch.load(spec['checkpoint'],weights_only=True,map_location=device));return h.to(device).eval()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cost-sweep',help='Selected existing candidate key, chosen on validation only');args=ap.parse_args()
 torch.set_num_threads(2);torch.manual_seed(20261044)
 assert read(RUN/'status.json')['phase']=='complete'
 m=read(RUN/'manifest.json');records=torch.load(RUN/'states.pt',weights_only=True);x=torch.stack([r['pack'] for r in records]).float();legal=torch.stack([r['legal'] for r in records]);weights=state_weights([r['qid'] for r in records],'cpu')
 assert set(r['qid'] for r in records)<=set(m['train'])
 base=PolicyHead();base.load_state_dict(torch.load(m['reference'],weights_only=True,map_location='cpu'));base.eval()
 with torch.no_grad():ref=distribution(base(x,legal)[0],legal).probs.detach()
 def fit(h,lam):
  q,quality,cost=action_values(records,lam);opt=torch.optim.AdamW([p for p in h.parameters() if p.requires_grad],lr=1e-4);curve=[]
  for epoch in range(100):
   opt.zero_grad();loss=expected_utility_loss(h(x,legal)[0],legal,q,ref,weights);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.);opt.step()
   if epoch%10==0 or epoch==99:curve.append({'epoch':epoch,'loss':float(loss.detach())})
  with torch.no_grad():
   p=distribution(h(x,legal)[0],legal).probs;kl=(p*(p.clamp_min(1e-9).log()-ref.clamp_min(1e-9).log())).sum(-1)
   metrics={'train_expected_Q_gain':float((weights*((p-ref)*q).sum(-1)).sum()),'train_kl':float((weights*kl).sum()),'train_changed_fraction':float((p.argmax(-1)!=ref.argmax(-1)).float().mean()),'curve':curve}
  return h.eval(),metrics
 def save_head(name,h,spec,metrics):
  checkpoint=OUT/f'{name}.pt';torch.save(h.state_dict(),checkpoint)
  return spec|{'checkpoint':str(checkpoint),'sha256':sha(checkpoint),'trainable_parameters':sum(p.numel() for p in h.parameters() if p.requires_grad),'metrics':metrics}
 if args.cost_sweep:
  manifest=read(OUT/'manifest.json');selected=manifest['candidates'][args.cost_sweep]
  for lam in [.01,.06]:
   init=manifest['candidates'][selected['warmstart_key']] if 'warmstart_key' in selected else manifest['candidates']['grpo_baseline']
   h=load_head(init)
   if init['kind']=='original':
    for n,p in h.named_parameters():p.requires_grad_(n.startswith(('action.','skip.')))
   name=args.cost_sweep+f'_lambda{lam:.2f}';assert name not in manifest['candidates']
   h,metrics=fit(h,lam);manifest['candidates'][name]=save_head(name,h,{k:v for k,v in selected.items() if k in ['kind','inputs','body','warmstart_key']}|{'lambda':lam},metrics)
  save(OUT/'manifest.json',manifest);return
 OUT.mkdir(parents=True,exist_ok=False)
 candidates={'grpo_baseline':{'kind':'original','checkpoint':m['reference'],'sha256':sha(Path(m['reference']))}}
 for n,p in base.named_parameters():p.requires_grad_(n.startswith(('action.','skip.')))
 h,metrics=fit(copy.deepcopy(base),.03);candidates['same_structure_improved']=save_head('same_structure_improved',h,{'kind':'original','lambda':.03},metrics)
 for inputs in ['last','multi']:
  for body in ['linear','mlp']:
   torch.manual_seed(20261044);h=StateHead(inputs=inputs,body=body);h.normalize_from_training(x);opt=torch.optim.AdamW(h.parameters(),lr=.001)
   for epoch in range(400):
    opt.zero_grad();p=distribution(h(x,legal)[0],legal).probs
    loss=(weights*(ref*(ref.clamp_min(1e-9).log()-p.clamp_min(1e-9).log())).sum(-1)).sum();loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.);opt.step()
   key=f'{inputs}_{body}';wk=key+'_warmstart';spec={'kind':'structure','inputs':inputs,'body':body}
   with torch.no_grad():
    p=distribution(h(x,legal)[0],legal).probs;kl=float((weights*(ref*(ref.clamp_min(1e-9).log()-p.clamp_min(1e-9).log())).sum(-1)).sum())
   candidates[wk]=save_head(wk,h,spec,{'train_distillation_KL':kl})
   h,metrics=fit(h,.03);candidates[key]=save_head(key,h,spec|{'lambda':.03,'warmstart_key':wk},metrics)
 manifest={'candidates':candidates,'train_states':len(records),'source_data_sha256':sha(RUN/'states.pt'),'train':m['train'],'validation':m['validation'],'test':m['test'],'fit_seed':20261044,'epochs':100,'warmstart_epochs':400,'lr':1e-4,'reference_kl':.02,'entropy':.01,'source_sha256':{p:sha(REPO/p) for p in ['scripts/fit_decision_state_heads.py','src/lsdrag/decision_state_learning.py']},'note':'full-action MC policy improvement, not on-policy GRPO; original trunk frozen, new structure parameters trained with common distillation warmstart'}
 save(OUT/'manifest.json',manifest);save(OUT/'status.json',{'phase':'complete','candidates':len(candidates),'pid':None});print(json.dumps({k:v.get('metrics') for k,v in candidates.items()},indent=2))
if __name__=='__main__':main()
