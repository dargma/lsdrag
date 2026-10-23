"""Inspect actual serialized heads and same-state output numerics on CPU."""
from pathlib import Path
import json,hashlib,numpy as np,torch
from lsdrag.decision_policy import PolicyHead
REPO=Path(__file__).resolve().parents[2];BASE=Path('/content/local_fast/lsdrag-text-v1/results')
def main():
 torch.set_num_threads(2);initial=torch.load(BASE/'decision_policy_20261023_pilot2/reference_round0.pt',weights_only=True,map_location='cpu');rows=torch.load(BASE/'decision_policy_20261023_pilot2/round0/counterfactual_data.pt',weights_only=True,map_location='cpu');x=torch.stack([r['pack'] for r in rows]).float();legal=torch.stack([r['legal'] for r in rows]);out={};baseprobs=None
 paths={'anchor':BASE/'decision_policy_20261023_pilot2/reference_round0.pt','ppo96':BASE/'decision_ppo96_20261023/ppo_iteration2.pt','rwr96':BASE/'decision_ppo96_20261023/rwr_iteration2.pt'}
 for name,path in paths.items():
  weights=torch.load(path,weights_only=True,map_location='cpu');h=PolicyHead(2560,'mlp');h.load_state_dict(weights);h.eval();layers={}
  for k,t in weights.items():
   v=t.float();delta=v-initial[k].float();layers[k]={'shape':list(t.shape),'finite':bool(torch.isfinite(v).all()),'norm':float(v.norm()),'max_abs':float(v.abs().max()),'delta_norm':float(delta.norm()),'delta_relative':float(delta.norm()/initial[k].float().norm().clamp_min(1e-12))}
  assert all(v['finite'] for v in layers.values());assert bool((weights['scale']>0).all())
  with torch.no_grad():logits,critic,_,_=h(x,legal);p=logits.softmax(-1)
  assert torch.isfinite(logits[legal]).all() and torch.isfinite(p).all() and torch.allclose(p.sum(-1),torch.ones(len(rows))) and bool((p[~legal]==0).all())
  if name=='anchor':baseprobs=p
  counts=torch.bincount(p.argmax(-1),minlength=4).tolist();trunk_unchanged=all(layers[k]['delta_norm']==0 for k in layers if k.startswith('trunk.') or k in ['mean','scale'])
  out[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'tensor_count':len(weights),'total_stored_elements':sum(t.numel() for t in weights.values()),'layers':layers,'trunk_normalizer_unchanged':trunk_unchanged,'policy_probe':{'states':len(rows),'logit_legal_min':float(logits[legal].min()),'logit_legal_max':float(logits[legal].max()),'pmax_median':float(p.max(-1).values.median()),'pmax_gt099_fraction':float((p.max(-1).values>.99).float().mean()),'entropy_mean':float(-(p*p.clamp_min(1e-30).log()).sum(-1).mean()),'action_counts':counts,'argmax_changed_vs_anchor':int((p.argmax(-1)!=baseprobs.argmax(-1)).sum()),'critic_min':float(critic.min()),'critic_max':float(critic.max())}}
 target=REPO/'lab/reports/decision_head_weight_audit_20261023';target.mkdir(parents=True,exist_ok=True);(target/'result.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=['# 실제 head weight 및 출력 검사','', '실제.pt파일을CPU로열어전체tensor shape/NaN/Inf/norm/최댓값/초기대비차이검사. 기존train67hidden에forward하여합법확률/합1/포화/행동분포확인. 품질이나이해능력검증을대체하지않는다.','', '| head | finite | trunk/normalizer 동일 | maxp 중앙값 | maxp>.99 | 행동별수 | anchor대비변경 |','|---|---|---|---:|---:|---|---:|']
 for name,r in out.items():
  p=r['policy_probe'];lines.append(f"| {name} | True | {r['trunk_normalizer_unchanged']} | {p['pmax_median']:.3f} | {p['pmax_gt099_fraction']:.3f} | {p['action_counts']} | {p['argmax_changed_vs_anchor']} |")
 lines+=['','행동순서keyword/semantic/read/FINAL. 금지logit−1e9는mask이므로폭주로해석하지않고legal값만통계. success/cost출력층은현재훈련대상아니어서초기값유지,성공확률로해석금지. critic은finite여도예측력이약할수있음: PPO96 on-policy explained variance는후기round에서약−.001/−.007이었다. Critic영향과actor가중치정상여부를구분한다.','', '## 학습층 변화','', '| head | tensor | norm | delta norm | max abs |','|---|---|---:|---:|---:|']
 for name,r in out.items():
  if name=='anchor':continue
  for k,v in r['layers'].items():
   if k.startswith(('action.','skip.','value.')):lines.append(f"| {name} | {k} | {v['norm']:.5f} | {v['delta_norm']:.5f} | {v['max_abs']:.5f} |")
 lines+=['','현재파일은손상/NaN/전체한행동붕괴같은명백한garbage징후가없다. Weight범위정상만으로학습의유용성을주장하지않으며실제heldout품질과함께판단한다. Critic은zero초기화했으므로표의anchor대비변화에는reset효과가포함됨.']
 (REPO/'docs/JEV_DECISION_HEAD_WEIGHT_AUDIT_20261023.md').write_text('\n'.join(lines)+'\n');print(json.dumps({n:{k:v for k,v in r.items() if k!='layers'} for n,r in out.items()},indent=2))
if __name__=='__main__':main()
