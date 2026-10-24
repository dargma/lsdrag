"""Explicit experimental action-prefix intervention, never a learned policy."""
import torch
from torch import nn
class PrefixIntervention(nn.Module):
    def __init__(self,anchor,actions):
        super().__init__();self.anchor=anchor;self.actions=tuple(actions);self.index=0;self.reference_probs=[]
    def forward(self,pack,legal):
        logits,value,qa,ca=self.anchor(pack,legal)
        self.reference_probs.append(logits.softmax(-1)[0].detach().cpu().tolist())
        if self.index<len(self.actions):
            action=self.actions[self.index]
            if not bool(legal[0,action]):raise ValueError('Frozen intervention action is unavailable')
            logits=torch.full_like(logits,-1e9);logits[:,action]=0
        self.index+=1
        return logits,value,qa,ca
