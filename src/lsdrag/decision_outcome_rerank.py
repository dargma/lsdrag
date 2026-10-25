"""Shared frozen state representation; outcome predictions are not policy probabilities."""
import torch
from torch import nn
from torch.nn import functional as F
from .decision_state_learning import StateHead
class OutcomeRerankHead(nn.Module):
 def __init__(self):
  super().__init__();self.base=StateHead(dim=4096,inputs='multi',body='mlp');self.base.requires_grad_(False)
  self.success=nn.Linear(128,4);self.logcost=nn.Linear(128,4);self.register_buffer('supported',torch.zeros(4,dtype=torch.bool))
 def predict(self,pack):
  z=self.base.net[1](self.base.net[0]((self.base.features(pack)-self.base.mean)/self.base.scale))
  return self.success(z).sigmoid(),F.softplus(self.logcost(z))
 def forward(self,pack,legal):
  logits=self.base(pack,legal)[0];q,lc=self.predict(pack);cost=lc.expm1();ref=logits.argmax(-1);rq=q.gather(1,ref[:,None]);rc=cost.gather(1,ref[:,None])
  eligible=legal&self.supported[None,:]&(q>=rq+.05)&(cost<=.90*rc)
  chosen=cost.masked_fill(~eligible,torch.inf).argmin(-1);chosen=torch.where(eligible.any(-1),chosen,ref)
  # Preserve original logits when no supported predicted quality+cost improvement exists.
  out=logits.clone();override=eligible.any(-1);out[override]=-1e9;out[override,chosen[override]]=0
  return out,out.new_zeros(len(out)),q,lc
