"""Cached-state policy learning with explicit all-action counterfactual outcomes.

This is a regularized policy-improvement surrogate, not on-policy GRPO.
No claim that finite-rollout Q targets are calibrated future-success estimates.
"""
import torch
from torch import nn
from .decision_policy import distribution

class StateHead(nn.Module):
    def __init__(self, dim=2560, inputs='last', body='mlp'):
        super().__init__()
        if inputs not in ('last','multi') or body not in ('linear','mlp'):
            raise ValueError('Unsupported structure')
        self.inputs,self.body,self.dim=inputs,body,dim
        width=dim*(3 if inputs=='multi' else 1)
        self.register_buffer('mean',torch.zeros(width));self.register_buffer('scale',torch.ones(width))
        self.net=nn.Linear(width,4) if body=='linear' else nn.Sequential(nn.Linear(width,128),nn.Tanh(),nn.Linear(128,4))
    def features(self,pack):
        return pack[:,0].float() if self.inputs=='last' else pack[:,[0,1,4]].float().flatten(1)
    def normalize_from_training(self,pack):
        with torch.no_grad():
            x=self.features(pack);self.mean.copy_(x.mean(0));self.scale.copy_(x.std(0,unbiased=False).clamp_min(.1))
    def forward(self,pack,legal):
        logits=self.net((self.features(pack)-self.mean)/self.scale).masked_fill(~legal,-1e9)
        zero=logits.new_zeros(len(logits));aux=logits.new_zeros(len(logits),4)
        return logits,zero,aux,aux

def action_values(records,cost_weight=.03):
    quality=torch.zeros(len(records),4);cost=torch.zeros_like(quality);legal=torch.stack([r['legal'] for r in records]).bool()
    for i,r in enumerate(records):
        for a in range(4):
            samples=[s for s in r['samples'] if s['action']==a]
            if legal[i,a] and not samples:raise ValueError('Missing legal action outcomes')
            if not legal[i,a] and samples:raise ValueError('Illegal action outcome')
            if samples:
                quality[i,a]=sum(s['f1'] for s in samples)/len(samples)
                cost[i,a]=sum(s['cost'] for s in samples)/len(samples)
    return quality-cost_weight*cost,quality,cost

def expected_utility_loss(logits,legal,q,reference,weights,kl_coef=.02,entropy_coef=.01):
    if not torch.isfinite(q[legal]).all():raise ValueError('Nonfinite legal Q')
    p=distribution(logits,legal).probs;logp=p.clamp_min(1e-9).log()
    safeq=q.masked_fill(~legal,0).detach()
    # State-dependent baseline cannot change the exact all-action gradient.
    baseline=(reference*safeq).sum(-1,keepdim=True)
    utility=(p*(safeq-baseline)).sum(-1)
    kl=(p*(logp-reference.clamp_min(1e-9).log())).sum(-1)
    entropy=-(p*logp).sum(-1)
    return (weights*(-utility+kl_coef*kl-entropy_coef*entropy)).sum()
