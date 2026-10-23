import torch
from torch import nn
from lsdrag.decision_policy_ablation import ProjectedHead,fit_global_bias
class Dummy(nn.Module):
 def __init__(self,values):super().__init__();self.register_buffer('values',torch.tensor(values,dtype=torch.float))
 def forward(self,p,l):return self.values.expand(len(p),-1).masked_fill(~l,-1e9),None,None,None

def test_stop_and_tool_projection():
 a=Dummy([3,1,2,2.5]);f=Dummy([1,4,2,3]);p=torch.zeros(2,1,1);legal=torch.tensor([[1,1,1,1],[0,0,0,1]],dtype=torch.bool)
 stop=ProjectedHead(a,f,'stop_only')(p,legal)[0];tool=ProjectedHead(a,f,'tool_only')(p,legal)[0]
 assert stop.argmax(-1).tolist()==[0,3]
 assert tool.argmax(-1).tolist()==[1,3]
 assert torch.allclose(stop[0,3]-stop[0,:3].max(),torch.tensor(-1.))
 assert torch.allclose(tool[0,3]-tool[0,:3].max(),torch.tensor(-.5))
 assert (stop[~legal]==-1e9).all()

def test_constant_bias_recovery():
 a=Dummy([0,0,0,0]);f=Dummy([0,1,-1,.5]);p=torch.zeros(4,1,1);legal=torch.ones(4,4,dtype=torch.bool)
 bias,stats=fit_global_bias(a,f,p,legal,['a','a','b','c'])
 assert stats['train_weighted_kl']<.001
 assert stats['train_weighted_argmax_agreement']==1
