"""Train-only saturation/gradient-responsibility audit; never opens test artifacts."""
from pathlib import Path
import json
import torch
from lsdrag.decision_policy import PolicyHead,distribution

torch.set_num_threads(2)
r=Path('/content/local_fast/lsdrag-text-v1/results/decision_policy_20261023_pilot1')
out=Path('lab/reports/decision_policy_20261023/pilot1_train_audit.json')
h=PolicyHead(2560,'mlp');h.load_state_dict(torch.load(r/'heads/last_mlp.pt',weights_only=True,map_location='cpu'))
f=torch.load(r/'train_features.pt',weights_only=True,map_location='cpu');x=f['pack'].float();legal=f['legal']
with torch.no_grad():logits=h(x,legal)[0]
temperatures={}
for t in [1,2,4,8,16]:
 p=distribution(logits/t,legal).probs
 temperatures[t]={'median_max_probability':float(p.max(-1).values.median()),'mean_entropy':float(distribution(logits/t,legal).entropy().mean())}
t=next((t for t,v in temperatures.items() if v['median_max_probability']<=.8),16)
iterations=[]
for it in range(3):
 h.load_state_dict(torch.load(r/('heads/bc_aux.pt' if it==0 else f'heads/ppo_iteration{it-1}.pt'),weights_only=True,map_location='cpu'))
 rows=torch.load(r/f'ppo/iteration{it}/on_policy_steps.pt',weights_only=True,map_location='cpu')
 x=torch.stack([s['pack'] for s in rows]).float();mask=torch.stack([s['legal'] for s in rows]);a=torch.tensor([s['action'] for s in rows])
 with torch.no_grad():z=h(x,mask)[0];p=distribution(z,mask).probs;mix=distribution(z,mask,.05).probs
 taken=p.gather(1,a[:,None]).squeeze();responsibility=.95*taken/mix.gather(1,a[:,None]).squeeze();non_greedy=a!=p.argmax(-1)
 adv=torch.tensor([s['return']-s['value'] for s in rows]);positive=non_greedy&(adv>0)
 iterations.append({'iteration':it,'n':len(rows),'median_max_probability':float(p.max(-1).values.median()),'mean_entropy':float(distribution(z,mask).entropy().mean()),'non_greedy_samples':int(non_greedy.sum()),'non_greedy_mean_actor_responsibility':float(responsibility[non_greedy].mean()) if non_greedy.any() else None,'positive_adv_non_greedy':int(positive.sum()),'positive_adv_non_greedy_responsibility':float(responsibility[positive].mean()) if positive.any() else None,'mean_abs_advantage':float(adv.abs().mean())})
report={'scope':'pilot1 training states and on-policy training rollouts only','temperature_grid_train_only':temperatures,'chosen_temperature':t,'iterations':iterations,'interpretation':'For epsilon-uniform mixture, gradient log p_mix(a) is scaled by (1-eps)*p_actor(a)/p_mix(a). Near-zero actor-probability exploratory actions therefore have weak actor gradient. This diagnoses a mechanism, not a causal ablation proving the sole cause.'}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
