"""Trainable post-reader representation and residual action policy, anchored at H3."""
import copy
import torch
from torch import nn
class SharedResidualPolicy(nn.Module):
    def __init__(self, anchor, hidden=2560, width=64):
        super().__init__()
        self.anchor=copy.deepcopy(anchor);self.anchor.requires_grad_(False)
        self.project=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,width),nn.Tanh())
        self.encoder=nn.Sequential(nn.Linear(width*4,width),nn.Tanh())
        self.actor=nn.Linear(width,4)
        nn.init.zeros_(self.actor.weight);nn.init.zeros_(self.actor.bias)
        self.register_buffer('strength',torch.tensor(1.))
    def forward(self,pack,legal):
        z=self.project(pack)
        attn=(z[:,5:]*z[:,1,None]).sum(-1)/(z.shape[-1]**.5)
        pool=(attn.softmax(-1)[...,None]*z[:,5:]).sum(1)
        h=self.encoder(torch.cat([pool,z[:,0],z[:,1],z[:,4]],-1))
        logits=(self.anchor(pack,legal)[0]+self.strength*self.actor(h)).masked_fill(~legal,-1e9)
        zero=logits.new_zeros(len(logits));aux=logits.new_zeros(len(logits),4)
        return logits,zero,aux,aux
