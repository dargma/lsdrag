import torch
import pytest
from lsdrag.decision_state_learning import StateHead
from lsdrag.decision_structure_adapter import StructureAdapter

@pytest.mark.parametrize('variant',['depth','input_gate','depth_input_gate','observation'])
def test_identity_updates_and_frozen_normalization(variant):
    torch.manual_seed(7);x=torch.randn(5,37,8);base=StateHead(dim=8,inputs='multi');base.normalize_from_training(x)
    mask=torch.ones(5,4,dtype=torch.bool);mask[:,2]=False
    h=StructureAdapter(base,variant);h.normalize_extra(x);assert torch.equal(base(x,mask)[0],h(x,mask)[0])
    mean=h.base.mean.clone();scale=h.base.scale.clone();opt=torch.optim.AdamW(h.parameters(),lr=.01)
    for _ in range(3):
        opt.zero_grad();loss=-h(x,mask)[0].log_softmax(-1)[:,1].mean();loss.backward();opt.step()
    assert torch.equal(mean,h.base.mean) and torch.equal(scale,h.base.scale)
    assert not torch.equal(base(x,mask)[0],h(x,mask)[0])
    assert (h(x,mask)[0].softmax(-1)[:,2]==0).all()
    parameter=h.up.weight if variant in ['depth','depth_input_gate'] else h.obs_up.weight if variant=='observation' else h.gate_out.weight
    assert parameter.abs().sum()>0
