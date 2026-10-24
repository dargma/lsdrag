import pytest
import torch
from lsdrag.decision_value_head import DecisionValueHead, stopping_opportunity_loss, value_prediction_loss

@pytest.mark.parametrize('body',['mlp','attention'])
def test_shapes_mask_gradients_and_serialization(body):
    torch.manual_seed(3)
    h=DecisionValueHead(dim=8,width=6,body=body)
    x=torch.randn(5,5,8);h.normalize_from_training(x)
    legal=torch.ones(5,4,dtype=torch.bool);legal[:,2]=False
    logits,_,q,c=h(x,legal)
    assert (logits.softmax(-1)[:,2]==0).all()
    assert ((q>=0)&(q<=1)).all() and (c>=0).all()
    loss=value_prediction_loss(q,c,legal,torch.rand(5,4),torch.rand(5,4)*4,torch.ones(5)/5)
    loss.backward()
    assert h.encoder.weight.grad.abs().sum()>0
    if body=='attention':
        _,w=h.representation(x)
        assert torch.allclose(w.sum(-1),torch.ones(5,4))
        assert h.action_query.grad.abs().sum()>0
    copy=DecisionValueHead(dim=8,width=6,body=body);copy.load_state_dict(h.state_dict())
    assert torch.equal(copy(x,legal)[0],logits)

@pytest.mark.parametrize('continue_better',[True,False])
def test_stop_ranking_pushes_correct_direction(continue_better):
    logits=torch.zeros(1,4,requires_grad=True)
    legal=torch.tensor([[True,False,False,True]])
    quality=torch.tensor([[.9,100.,100.,.1]]) if continue_better else torch.tensor([[.1,100.,100.,.9]])
    loss=stopping_opportunity_loss(logits,legal,quality,torch.zeros(1,4),torch.ones(1))
    loss.backward()
    assert (logits.grad[0,0]<0).item()==continue_better
    assert logits.grad[0,1]==0 and logits.grad[0,2]==0


def test_all_failed_and_final_only_have_no_ranking_gradient():
    logits=torch.zeros(2,4,requires_grad=True)
    legal=torch.tensor([[True,True,False,True],[False,False,False,True]])
    loss=stopping_opportunity_loss(logits,legal,torch.zeros(2,4),torch.rand(2,4),torch.ones(2)/2)
    loss.backward()
    assert loss.item()==0 and torch.equal(logits.grad,torch.zeros_like(logits))
