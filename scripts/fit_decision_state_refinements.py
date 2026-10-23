"""Three validation-driven, predeclared refinements for the raised joint target."""
import copy,json
from pathlib import Path
import torch
from run_decision_policy_pilot import ROOT,REPO,save,sha
from fit_decision_state_heads import load_head
from lsdrag.decision_policy import distribution
from lsdrag.decision_policy_followup import state_weights
from lsdrag.decision_state_learning import action_values,expected_utility_loss
RUN=ROOT/'results/decision_state_outcomes_20261023';OUT=ROOT/'results/decision_state_heads_20261023'
SPECS=[('multi_mlp_cost006','multi_mlp_warmstart','proxy',.06),('multi_mlp_generation015','multi_mlp_warmstart','generated',.15),('last_linear_cost009','last_linear_warmstart','proxy',.09)]
def main():
 torch.set_num_threads(2);torch.manual_seed(20261044);m=json.loads((OUT/'manifest.json').read_text());rows=torch.load(RUN/'states.pt',weights_only=True);x=torch.stack([r['pack'] for r in rows]).float();legal=torch.stack([r['legal'] for r in rows]);w=state_weights([r['qid'] for r in rows],'cpu');baseline=load_head(m['candidates']['grpo_baseline'])
 with torch.no_grad():ref=distribution(baseline(x,legal)[0],legal).probs.detach()
 _,quality,proxy=action_values(rows,0.)
 gen=torch.zeros_like(proxy)
 for i,r in enumerate(rows):
  for a in range(4):
   samples=[s['generated']/100 for s in r['samples'] if s['action']==a]
   if samples:gen[i,a]=sum(samples)/len(samples)
 for name,warm,kind,coef in SPECS:
  assert name not in m['candidates'];h=load_head(m['candidates'][warm]);opt=torch.optim.AdamW(h.parameters(),lr=1e-4);q=quality-coef*(proxy if kind=='proxy' else gen)
  for epoch in range(100):
   opt.zero_grad();loss=expected_utility_loss(h(x,legal)[0],legal,q,ref,w);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.);opt.step()
  checkpoint=OUT/f'{name}.pt';torch.save(h.state_dict(),checkpoint);spec=m['candidates'][warm]
  m['candidates'][name]={k:spec[k] for k in ['kind','inputs','body']}|{'checkpoint':str(checkpoint),'sha256':sha(checkpoint),'warmstart_key':warm,'lambda':coef,'cost_mode':kind,'epochs':100,'trainable_parameters':sum(p.numel() for p in h.parameters()),'training_data_sha256':sha(RUN/'states.pt'),'source_sha256':sha(Path(__file__))}
 save(OUT/'manifest.json',m);print([s[0] for s in SPECS])
if __name__=='__main__':main()
