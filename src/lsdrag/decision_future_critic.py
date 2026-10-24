"""Small post-reader representation and masked Monte Carlo action-value heads."""
import torch
from torch import nn
from torch.nn import functional as F

class FutureCritic(nn.Module):
    def __init__(self, hidden=2560, width=64):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden,width),nn.Tanh())
        self.encoder = nn.Sequential(nn.Linear(4*width,width),nn.Tanh())
        self.quality = nn.Linear(width,4)
        self.cost = nn.Linear(width,4)
        self.actor = nn.Linear(width,4)
        nn.init.zeros_(self.actor.weight); nn.init.zeros_(self.actor.bias)
        self.actor.requires_grad_(False)
    def forward(self, pack):
        z=self.project(pack)
        attention=(z[:,5:] * z[:,1,None]).sum(-1)/(z.shape[-1]**.5)
        pooled=(attention.softmax(-1)[...,None]*z[:,5:]).sum(1)
        h=self.encoder(torch.cat([pooled,z[:,0],z[:,1],z[:,4]],-1))
        return self.quality(h).sigmoid(),F.softplus(self.cost(h))*100,self.actor(h)

def masked_loss(q,c,target_q,target_c,legal,weights):
    # where BEFORE arithmetic also prevents excluded NaN labels from poisoning gradients.
    qerr=(q-torch.where(legal,target_q,q.detach())).square()
    cerr=((c-torch.where(legal,target_c,c.detach()))/100).square()
    return (((qerr+cerr).sum(-1)/legal.sum(-1))*weights).sum()
