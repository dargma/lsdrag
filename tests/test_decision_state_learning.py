import pytest
import torch
from lsdrag.decision_state_learning import StateHead,action_values,expected_utility_loss

def test_expected_gradient_rewards_better_action_and_ignores_illegal_values():
    z=torch.zeros(1,4,requires_grad=True);legal=torch.tensor([[True,True,False,False]])
    q=torch.tensor([[1.,0.,float('nan'),float('nan')]]);ref=torch.tensor([[.5,.5,0.,0.]])
    loss=expected_utility_loss(z,legal,q,ref,torch.ones(1),0.,0.);loss.backward()
    assert z.grad[0,0]<0 and z.grad[0,1]>0
    assert torch.equal(z.grad[0,2:],torch.zeros(2))

def test_state_constant_shift_preserves_gradient():
    legal=torch.ones(2,4,dtype=torch.bool);ref=torch.full((2,4),.25);q=torch.randn(2,4)
    grads=[]
    for values in (q,q+torch.tensor([[100.],[-12.]])):
        z=torch.zeros(2,4,requires_grad=True)
        expected_utility_loss(z,legal,values,ref,torch.full((2,),.5)).backward();grads.append(z.grad)
    assert torch.allclose(*grads,atol=1e-6)

def test_public_features_exclude_unused_slots_and_normalization_is_frozen():
    torch.manual_seed(1);x=torch.randn(3,37,8);legal=torch.ones(3,4,dtype=torch.bool)
    for inputs in ('last','multi'):
        h=StateHead(8,inputs,'mlp');h.normalize_from_training(x);old=h.mean.clone();z=h(x,legal)[0]
        changed=x.clone();changed[:,5:]+=1000
        assert torch.equal(z,h(changed,legal)[0]);assert torch.equal(h.mean,old)

def test_missing_action_rejected():
    with pytest.raises(ValueError,match='Missing'):
        action_values([{'legal':torch.tensor([True,True,False,False]),'samples':[{'action':0,'f1':1,'cost':1}]}])
