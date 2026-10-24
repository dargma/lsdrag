import torch
from lsdrag.decision_relative_critic import relative_quality_loss

def test_offsets_and_unobserved_targets_do_not_change_learning():
    q=torch.tensor([[.2,.6,.8,.1],[.4,.5,.2,.7]],requires_grad=True)
    c=torch.full_like(q,30.)
    legal=torch.tensor([[True,True,False,True],[True,False,True,True]])
    target=torch.tensor([[1.,.5,float('nan'),0.],[0.,float('nan'),1.,.5]])
    cost=torch.where(legal,torch.full_like(target,20.),torch.full_like(target,float('nan')))
    weights=torch.tensor([.5,.5])
    first=relative_quality_loss(q,c,target,cost,legal,weights)
    second=relative_quality_loss(q+torch.tensor([[.13],[-.4]]),c,target+torch.tensor([[7.],[-9.]]),cost,legal,weights)
    assert torch.allclose(first,second)
    first.backward()
    assert torch.isfinite(q.grad).all() and (q.grad[~legal]==0).all()
    exact=relative_quality_loss(torch.nan_to_num(target),torch.full_like(cost,20),target,cost,legal,weights)
    assert exact.item()==0
