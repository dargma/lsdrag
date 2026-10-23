"""Offline action-value ridge baseline; not an online LinUCB reproduction."""
import torch
from torch import nn
from .decision_policy_followup import state_weights

class RidgeValueHead(nn.Module):
    def __init__(self,mean,scale):
        super().__init__();d=mean.numel()
        self.register_buffer('mean',mean.detach().clone());self.register_buffer('scale',scale.detach().clone())
        self.register_buffer('weight',torch.zeros(4,d,device=mean.device))
        self.register_buffer('bias',torch.zeros(4,device=mean.device))
    def forward(self,pack,legal):
        x=(pack[:,0].float()-self.mean)/self.scale
        q=x@self.weight.T+self.bias
        return q.masked_fill(~legal,-1e9),q.masked_fill(~legal,-1e9).max(-1).values,torch.zeros_like(q),torch.zeros_like(q)

def fit_ridge(head,x,legal,targets,qids,penalty=1.):
    """Question-balanced weighted squared error + penalty||w||²; free intercept.

    Dual solve avoids a hidden_dim² covariance with fewer than 200 states.
    Fit each action only on its observed legal states. No evaluation targets.
    """
    if penalty<=0:raise ValueError('penalty must be positive')
    z=((x[:,0].float()-head.mean)/head.scale).double()
    stats=[]
    for a in range(4):
        idx=legal[:,a].nonzero().flatten()
        if not len(idx):raise ValueError('action lacks training coverage')
        za=z[idx];y=targets[idx,a].double();ids=[qids[i] for i in idx.tolist()]
        w=state_weights(ids,z.device).double();w=w/w.sum()
        xm=(w[:,None]*za).sum(0);ym=(w*y).sum();xc=za-xm;yc=y-ym
        X=xc*w.sqrt()[:,None];Y=yc*w.sqrt()
        coef=X.T@torch.linalg.solve(X@X.T+penalty*torch.eye(len(idx),dtype=torch.float64,device=z.device),Y)
        intercept=ym-xm@coef
        head.weight[a].copy_(coef.float());head.bias[a].copy_(intercept.float())
        stats.append({'action':a,'states':len(idx),'weighted_mse':float((w*(za@coef+intercept-y).square()).sum())})
    return head,stats
