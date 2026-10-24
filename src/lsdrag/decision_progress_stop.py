"""Causal exploration-progress residual sharing the frozen H3 reader representation."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

def progress_features(current, previous, previous_action, turn):
    if previous is None:
        assert turn==0
        return current.new_zeros(current.shape[:-1]+(current.shape[-1]+4,))
    assert turn>0 and 0<=previous_action<3
    action=F.one_hot(torch.tensor([previous_action],device=current.device),3).to(current.dtype)
    step=current.new_full((1,1),turn/8)
    return torch.cat([current-previous,action,step],dim=-1)

class ProgressStopPolicy(nn.Module):
    def __init__(self,anchor):
        super().__init__()
        self.anchor=copy.deepcopy(anchor).requires_grad_(False)
        self.progress=nn.Linear(anchor.base.net[0].out_features+4,1,bias=False)
        nn.init.zeros_(self.progress.weight)
        self.reset_episode()
    def reset_episode(self):
        # Transient causal cache; never saved as learned parameters.
        self.previous=None
        self.previous_action=None
        self.turn=0
    def state(self,pack):
        base=self.anchor.base
        return base.net[1](base.net[0]((base.features(pack)-base.mean)/base.scale))
    def forward(self,pack,legal):
        assert pack.shape[0]==1,'Online causal state requires one episode at a time'
        z=self.state(pack)
        features=progress_features(z,self.previous,self.previous_action,self.turn)
        logits,value,qa,ca=self.anchor(pack,legal)
        logits=logits.clone();logits[:,3]+=self.progress(features).squeeze(-1)
        logits=logits.masked_fill(~legal,-1e9)
        self.previous=z.detach().clone()
        self.previous_action=int(logits.argmax(-1))
        self.turn+=1
        return logits,value,qa,ca
