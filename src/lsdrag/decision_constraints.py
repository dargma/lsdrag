"""Offline state-weighted quality/token constraints; not a closed-loop guarantee."""
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from .decision_policy import distribution

def constrained_loss(logits,legal,quality,generated,weights,reference,budget,quality_floor,dual_tokens,dual_quality):
    if budget<=0:raise ValueError('Token budget must be positive')
    p=distribution(logits,legal).probs
    q=(weights*(p*quality.detach().masked_fill(~legal,0)).sum(-1)).sum()
    g=(weights*(p*generated.detach().masked_fill(~legal,0)).sum(-1)).sum()
    token_violation=g/budget-1
    quality_violation=(quality_floor-q)/.05
    logp=p.clamp_min(1e-9).log()
    kl=(weights*(p*(logp-reference.clamp_min(1e-9).log())).sum(-1)).sum()
    entropy=-(weights*(p*logp).sum(-1)).sum()
    loss=-q+dual_tokens*token_violation+dual_quality*quality_violation+.02*kl-.005*entropy
    return loss,q,g,token_violation,quality_violation

def feasibility(quality,generated,legal,weights,budget,quality_floor):
    """Optimistic unrestricted randomized policy upper bound on the frozen cache."""
    q,g,l,w=[x.detach().cpu().numpy() for x in [quality,generated,legal,weights]]
    rows,acts=np.where(l);count=len(rows)
    eq=coo_matrix((np.ones(count),(rows,np.arange(count))),shape=(len(q),count)).tocsr()
    result=linprog(-w[rows]*q[rows,acts],A_ub=(w[rows]*g[rows,acts])[None,:],b_ub=[budget],A_eq=eq,b_eq=np.ones(len(q)),bounds=(0,1),method='highs')
    return {'solver_success':bool(result.success),'message':result.message,'quality_upper_bound':float(-result.fun) if result.success else None,'quality_floor_feasible':bool(result.success and -result.fun>=quality_floor-1e-8),'note':'Optimistic unrestricted per-state randomized policy on noisy fixed-BC outcomes. Not neural realizability or actual episode-token feasibility.'}
