"""Minimal workflow comparator: semantic search exactly once, then FINAL."""
import torch
from torch import nn

class SemanticOnceHead(nn.Module):
    def __init__(self):
        super().__init__();self.step=0
    def reset_episode(self):
        self.step=0
    def forward(self,pack,legal):
        action=1 if self.step==0 else 3
        assert legal[:,action].all(), 'Fixed rule action must be legal'
        self.step+=1
        logits=pack.new_full((len(pack),4),-1e9,dtype=torch.float32);logits[:,action]=0
        return logits,logits.new_zeros(len(pack)),logits.new_zeros(len(pack),4),logits.new_zeros(len(pack),4)
