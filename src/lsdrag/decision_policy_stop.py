"""STOP/CONTINUE adaptation with immutable historical BC tool selection."""
import copy
import torch
from torch import nn

class StopOnlyHead(nn.Module):
    def __init__(self, historical, initialization, temperature=8.):
        super().__init__()
        self.temperature=temperature
        self.register_buffer('tool_weight',historical['action_weight'].detach().clone())
        self.register_buffer('tool_bias',historical['action_bias'].detach().clone())
        self.register_buffer('mean',initialization.mean.detach().clone())
        self.register_buffer('scale',initialization.scale.detach().clone())
        self.trunk=copy.deepcopy(initialization.trunk)
        self.action=nn.Linear(128,2,device=self.tool_weight.device)
        self.skip=nn.Linear(self.tool_weight.shape[1],2,device=self.tool_weight.device)
        for layer in [self.action,self.skip]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,pack,legal):
        raw=pack[:,0].float()@self.tool_weight.T+self.tool_bias
        tools=raw[:,:3].masked_fill(~legal[:,:3],-1e9)
        best,chosen=tools.max(-1)
        x=(pack[:,0].float()-self.mean)/self.scale
        binary=torch.stack([best,raw[:,3]],-1)/self.temperature+self.action(self.trunk(x))+self.skip(x)
        binary=torch.stack([binary[:,0].masked_fill(~legal[:,:3].any(-1),-1e9),binary[:,1].masked_fill(~legal[:,3],-1e9)],-1)
        logits=torch.full_like(raw,-1e9)
        logits=logits.scatter(1,chosen[:,None],binary[:,:1])
        logits[:,3]=binary[:,1]
        logits=logits.masked_fill(~legal,-1e9)
        z=torch.zeros(pack.shape[0],device=pack.device)
        return logits,z,torch.zeros_like(logits),torch.zeros_like(logits)
