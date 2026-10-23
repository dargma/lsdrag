"""Late-state counterfactual policy improvement; not on-policy PPO."""
import copy
import math
import torch
from torch.nn import functional as F
from .decision_policy import distribution
from .text_decision_benchmark import canonical


def tempered_initialization(head, temperature):
    if temperature < 1:raise ValueError('Temperature must be >=1')
    h=copy.deepcopy(head)
    with torch.no_grad():
        for name,p in h.named_parameters():
            if name.startswith(('action.','skip.')):p.div_(temperature)
    return h


def state_snapshots(question,prompt,trajectory):
    """First/middle/last controllable decision, including before voluntary FINAL.

    Reconstruct only already-observed public tool outputs; cap-imposed FINAL is
    excluded from the controllable state set. No answer/gold field is accepted.
    """
    count=min(len(trajectory)+1,8)
    indices=sorted({0,(count-1)//2,count-1})
    result=[]
    for turn in indices:
        messages=[{'role':'system','content':prompt},{'role':'user','content':question}]
        for i,step in enumerate(trajectory[:turn]):
            tc={'id':f'call_{i}','type':'function','function':{'name':step['tool_name'],'arguments':canonical(step['arguments'])}}
            messages.extend([{'role':'assistant','content':None,'tool_calls':[tc]}, {'role':'tool','tool_call_id':tc['id'],'content':canonical(step['result'])}])
        result.append({'turn':turn,'messages':messages,'history':copy.deepcopy(trajectory[:turn])})
    return result


def state_weights(qids,device):
    counts={q:qids.count(q) for q in set(qids)}
    w=torch.tensor([1/counts[q] for q in qids],device=device,dtype=torch.float32)
    return w/w.sum()


def improvement_loss(logits, legal, q, anchor_probs, weights, objective, kl_coef=.02):
    p=distribution(logits,legal).probs
    logp=p.clamp_min(1e-9).log()
    kl=(p*(logp-anchor_probs.clamp_min(1e-9).log())).sum(-1)
    entropy=-(p*logp).sum(-1)
    safeq=q.masked_fill(~legal,0)
    if objective=='return_pg':
        # All-action Monte Carlo cost-to-go under the frozen reference policy.
        # Subtract a per-state constant to reduce numerical scale, not a label.
        baseline=(safeq*legal).sum(-1)/legal.sum(-1)
        loss=-(p*(safeq-baseline[:,None])).sum(-1)
    elif objective=='outcome_ce':
        target=(q.masked_fill(~legal,-1e9)/.1).softmax(-1)
        loss=-(target*logp).sum(-1)
    else:raise ValueError(objective)
    return (weights*(loss+kl_coef*kl-.01*entropy)).sum(),(weights*kl).sum()


def fit_improvement(initial,anchor,x,legal,q,qids,objective,epochs=100):
    head=copy.deepcopy(initial).train();weights=state_weights(qids,x.device)
    with torch.no_grad():anchor_probs=distribution(anchor(x,legal)[0],legal).probs
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    curve=[]
    for epoch in range(epochs):
        optimizer.zero_grad();logits,*_=head(x,legal)
        loss,kl=improvement_loss(logits,legal,q,anchor_probs,weights,objective)
        if not torch.isfinite(loss):raise ValueError('Nonfinite policy objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
        curve.append({'epoch':epoch,'loss':float(loss.detach()),'anchor_kl':float(kl.detach())})
    return head.eval(),curve
