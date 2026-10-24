"""No-retrieval baseline using the same FINAL generation path."""
import torch
from torch import nn

class FinalOnlyHead(nn.Module):
    def forward(self,pack,legal):
        assert legal[:,3].all()
        logits=pack.new_full((len(pack),4),-1e9,dtype=torch.float32)
        logits[:,3]=0
        return logits,logits.new_zeros(len(pack)),logits.new_zeros(len(pack),4),logits.new_zeros(len(pack),4)
