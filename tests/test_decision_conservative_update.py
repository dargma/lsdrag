import copy
import torch
from lsdrag.decision_state_learning import StateHead
from lsdrag.decision_stop_value import StopValueHead
from lsdrag.decision_conservative_update import conservative_output_update

def setup():
 torch.manual_seed(7)
 base=StopValueHead(StateHead(dim=8,inputs='multi',body='mlp'));head=copy.deepcopy(base)
 x=torch.randn(16,37,8);legal=torch.ones(16,4,dtype=torch.bool)
 with torch.no_grad():head.base.net[2].bias[3]+=8
 return base,head,x,legal

def test_backtracking_enforces_training_bounds_and_preserves_reference():
 base,head,x,legal=setup();snapshot={k:v.clone() for k,v in base.state_dict().items()}
 mask=torch.ones(16,dtype=torch.bool)
 report=conservative_output_update(head,base,x,legal,[str(i//2) for i in range(16)],mask,change_limit=0.)
 assert report['accepted'] and 0<report['alpha']<1
 assert report['kl']<=.01 and report['unsupported_change_fraction']==0
 assert all(torch.equal(v,snapshot[k]) for k,v in base.state_dict().items())
 assert all(torch.equal(v,snapshot[k]) for k,v in head.state_dict().items() if not k.startswith('base.net.2.'))

def test_rejected_update_restores_reference_exactly():
 base,head,x,legal=setup()
 report=conservative_output_update(head,base,x,legal,[str(i) for i in range(16)],torch.ones(16,dtype=torch.bool),kl_limit=0.,change_limit=0.,max_backtracks=1)
 assert not report['accepted'] and report['alpha']==0
 assert all(torch.equal(v,base.state_dict()[k]) for k,v in head.state_dict().items())
