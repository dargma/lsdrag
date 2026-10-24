"""Restrict measured current-policy improvement to stop/continue with EM protection."""
import copy
import torch
from lsdrag.decision_current_policy_targets import current_policy_targets

def safe_stop_targets(records):
    reduced=[]
    for r in records:
        row=copy.copy(r)
        samples={s['action']:s for s in r['samples']}
        anchor=max(samples,key=lambda a:r['reference_probs'][a])
        cont=max((a for a in samples if a!=3),key=lambda a:r['reference_probs'][a])
        assert anchor in (cont,3)
        actions=[a for a in (cont,3) if a==anchor or samples[a]['em']>=samples[anchor]['em']]
        row['legal']=torch.tensor([a in actions for a in range(4)])
        row['samples']=[samples[a] for a in actions]
        reduced.append(row)
    labels,evidence=current_policy_targets(reduced)
    for i,a in labels.items():
        r=records[i];samples={s['action']:s for s in r['samples']};ref=max(samples,key=lambda a:r['reference_probs'][a])
        assert samples[a]['f1']>=samples[ref]['f1'] and samples[a]['em']>=samples[ref]['em']
    evidence['scope']='Binary H3-suffix targets: per-state F1 and EM nondecrease, aggregate cost cap; no episode guarantee.'
    return labels,evidence
