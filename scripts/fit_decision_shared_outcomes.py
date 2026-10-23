"""Same-initialization shared-corpus all-action policy learning; two cost weights."""
import copy,json
import torch
from run_decision_policy_pilot import ROOT,REPO,sha,save
from fit_decision_state_heads import load_head
from lsdrag.decision_state_learning import action_values,expected_utility_loss
from lsdrag.decision_policy import distribution
from lsdrag.decision_policy_followup import state_weights
RUN=ROOT/'results/decision_shared_outcomes192_20261023';OUT=ROOT/'results/decision_shared_heads_20261023'
def main():
 torch.set_num_threads(2);torch.manual_seed(20261052);m=json.loads((RUN/'manifest.json').read_text());assert json.loads((RUN/'status.json').read_text())['phase']=='complete'
 rows=torch.load(RUN/'states.pt',weights_only=True);x=torch.stack([r['pack'] for r in rows]).float();legal=torch.stack([r['legal'] for r in rows]);w=state_weights([r['qid'] for r in rows],'cpu');initial=load_head(m['reference_spec']);hm={'candidates':{'shared_bc':m['reference_spec']}};OUT.mkdir(parents=True,exist_ok=False)
 with torch.no_grad():ref=distribution(initial(x,legal)[0],legal).probs.detach()
 for lam in [.03,.06]:
  name=f'policy_iteration_cost{int(lam*100):03d}';assert name not in hm['candidates'];h=copy.deepcopy(initial)
  if m['reference_spec']['kind']=='original':
   for n,p in h.named_parameters():p.requires_grad_(n.startswith(('action.','skip.')))
  opt=torch.optim.AdamW([p for p in h.parameters() if p.requires_grad],lr=1e-4);q,_,_=action_values(rows,lam);curve=[]
  for epoch in range(100):
   opt.zero_grad();loss=expected_utility_loss(h(x,legal)[0],legal,q,ref,w);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.);opt.step()
   if epoch%10==0 or epoch==99:curve.append({'epoch':epoch,'loss':float(loss.detach())})
  cp=OUT/f'{name}.pt';torch.save(h.state_dict(),cp);hm['candidates'][name]={k:v for k,v in m['reference_spec'].items() if k in ['kind','inputs','body']}|{'checkpoint':str(cp),'sha256':sha(cp),'lambda':lam,'cost_mode':'proxy','initial_policy':m['reference_name'],'data_sha256':sha(RUN/'states.pt'),'train_questions':len(m['train']),'epochs':100,'lr':1e-4,'curve':curve}
 save(OUT/'manifest.json',hm);save(RUN/'fit_summary.json',{'candidates':['policy_iteration_cost003','policy_iteration_cost006'],'states':len(rows),'train_questions':len(m['train']),'test_evaluated':False})
if __name__=='__main__':main()
