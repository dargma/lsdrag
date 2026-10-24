"""Optional head-only context-bin access with exact policy-preserving initialization.

Bins are pooled public-prefix spans, not individual tokens or future information.
"""
import copy
import torch
from torch import nn

class ContextResidualHead(nn.Module):
    def __init__(self, base, width=64):
        super().__init__()
        self.base=copy.deepcopy(base).requires_grad_(False)
        dim=base.dim;self.width=width
        self.register_buffer('bin_mean',torch.zeros(32,dim))
        self.register_buffer('bin_scale',torch.ones(32,dim))
        self.keys=nn.Linear(dim,width,bias=False)
        self.values=nn.Linear(dim,width,bias=False)
        self.query=nn.Linear(dim,width,bias=False)
        self.action_queries=nn.Parameter(torch.empty(4,width))
        nn.init.normal_(self.action_queries,std=.02)
        self.positions=nn.Parameter(torch.zeros(32,width))
        # Zero final projection makes all initial logits identical to the learned base.
        self.output_weight=nn.Parameter(torch.zeros(4,width))

    def normalize_from_training(self,pack):
        if pack.shape[1]!=37:raise ValueError('Expected37-slot public-prefix pack')
        with torch.no_grad():
            x=pack[:,5:37].float()
            self.bin_mean.copy_(x.mean(0));self.bin_scale.copy_(x.std(0,unbiased=False).clamp_min(.1))

    def context(self,pack):
        if pack.shape[1]!=37:raise ValueError('Expected37-slot public-prefix pack')
        bins=(pack[:,5:37].float()-self.bin_mean)/self.bin_scale
        question=(pack[:,1].float()-self.base.mean[1])/self.base.scale[1]
        queries=self.query(question)[:,None,:]+self.action_queries
        keys=self.keys(bins)+self.positions
        attention=torch.einsum('bah,bsh->bas',queries,keys).div(self.width**.5).softmax(-1)
        return attention@self.values(bins),attention

    def forward(self,pack,legal):
        logits,value,quality,cost=self.base(pack,legal)
        context,_=self.context(pack)
        residual=(context*self.output_weight).sum(-1)
        return (logits+residual).masked_fill(~legal,-1e9),value,quality,cost
