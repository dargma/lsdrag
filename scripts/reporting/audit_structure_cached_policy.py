"""Train-only diagnostic; association is not causal or heldout policy evidence."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from evaluate_decision_attention import load_head
from lsdrag.decision_state_learning import action_values,expected_utility_loss
from lsdrag.decision_policy_followup import state_weights
ROOT=Path('/content/local_fast/lsdrag-text-v1/results')

def main():
 torch.set_num_threads(2)
 plan=json.loads(Path('lab/registry/decision_attention_20261024/plan.json').read_text())
 spec=json.loads(Path('lab/registry/decision_attention_20261024/validation_spec.json').read_text())
 records=torch.load(plan['outcomes'],weights_only=True,map_location='cpu')
 assert {r['qid'] for r in records}==set(plan['train'])
 x=torch.stack([r['pack'] for r in records]).float();legal=torch.stack([r['legal'] for r in records])
 q,quality,cost=action_values(records,.03);w=state_weights([r['qid'] for r in records],'cpu')
 ref=load_head(spec['policies']['selected_reference'])(x,legal)[0].softmax(-1).detach()
 generator=torch.Generator().manual_seed(20261062)
 permutations=[]
 # Shuffle between states within legal-mask groups; no invalid action confound.
 for _ in range(100):
  perm=torch.arange(len(records))
  for mask in legal.unique(dim=0):
   indices=(legal==mask).all(-1).nonzero().flatten()
   perm[indices]=indices[torch.randperm(len(indices),generator=generator)]
  permutations.append(perm)
 results={}
 with torch.no_grad():
  for name,checkpoint in spec['policies'].items():
   h=load_head(checkpoint);logits=h(x,legal)[0];p=logits.softmax(-1)
   utility=float((w*(p*q).sum(-1)).sum())
   shuffled=[float((w*(p[perm]*q).sum(-1)).sum()) for perm in permutations]
   entropy=-(p*p.clamp_min(1e-9).log()).sum(-1)
   results[name]={'common_training_loss':float(expected_utility_loss(logits,legal,q,ref,w)),
    'weighted_action_probabilities':(w[:,None]*p).sum(0).tolist(),
    'weighted_entropy':float((w*entropy).sum()),'expected_cached_q':utility,
    'shuffled_expected_q_mean':float(np.mean(shuffled)),
    'shuffled_expected_q_range':[min(shuffled),max(shuffled)],
    'state_alignment_above_shuffled':utility-float(np.mean(shuffled)),
    'expected_cached_f1':float((w*(p*quality).sum(-1)).sum()),
    'expected_cached_mixed_cost':float((w*(p*cost).sum(-1)).sum())}
 out=Path('lab/reports/decision_attention_20261024');out.mkdir(parents=True,exist_ok=True)
 report={'scope':'TRAIN only, 377 states/192 questions. Learned probabilities matched to old-policy 2-suffix/action Q. Shuffling diagnoses in-sample association, NOT causality or generalization. No threshold or model filtering from this audit.','seed':20261062,'permutations':100,'outcomes_sha256':hashlib.sha256(Path(plan['outcomes']).read_bytes()).hexdigest(),'results':results}
 (out/'cached_policy_audit.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# Train-only structure policy diagnostic','','No held-out or causal claim. No candidate filtering. Action order: keyword, semantic, read, FINAL.','', '| Policy | Common train loss | Matched Q | Q above shuffled | Entropy |','|---|---:|---:|---:|---:|']
 for name,r in results.items():lines.append(f"| {name} | {r['common_training_loss']:.5f} | {r['expected_cached_q']:.5f} | {r['state_alignment_above_shuffled']:.5f} | {r['weighted_entropy']:.4f} |")
 (out/'cached_policy_audit.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
