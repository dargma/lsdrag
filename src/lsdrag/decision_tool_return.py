"""Outcome-trained tool ranking with the frozen H3 greedy stop boundary."""
import copy
import torch
from torch import nn

class ToolReturnPolicy(nn.Module):
    def __init__(self, anchor, constant=False):
        super().__init__()
        self.anchor=copy.deepcopy(anchor).requires_grad_(False)
        self.constant=constant
        self.delta=nn.Linear(1 if constant else 128,3)
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
        if constant:self.delta.weight.requires_grad_(False)

    def features(self,pack):
        base=self.anchor.base
        x=(base.features(pack)-base.mean)/base.scale
        return base.net[1](base.net[0](x))

    def tool_logits(self,pack,legal):
        original=self.anchor(pack,legal)[0]
        z=original.new_zeros((len(pack),1)) if self.constant else self.features(pack)
        tools=(original[:,:3]+self.delta(z)).masked_fill(~legal[:,:3],-1e9)
        return tools,original

    def forward(self,pack,legal):
        tools,original=self.tool_logits(pack,legal)
        # Set maximum legal tool logit exactly to H3's maximum, preserving its
        # FINAL versus continue greedy boundary at every identical state.
        tools=tools-tools.max(-1,keepdim=True).values+original[:,:3].max(-1,keepdim=True).values
        logits=torch.cat((tools,original[:,3:]),-1).masked_fill(~legal,-1e9)
        zero=logits.new_zeros(len(logits));aux=logits.new_zeros(len(logits),4)
        return logits,zero,aux,aux
