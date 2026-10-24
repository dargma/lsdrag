import pytest
import torch
from lsdrag.decision_value_head import DecisionValueHead
from lsdrag.decision_context_residual import ContextResidualHead

def test_initial_policy_exact_and_learning_uses_context_without_changing_base():
    torch.manual_seed(9)
    x=torch.randn(6,37,8)
    base=DecisionValueHead(dim=8,width=6,body='mlp');base.normalize_from_training(x)
    head=ContextResidualHead(base,width=4);head.normalize_from_training(x)
    legal=torch.ones(6,4,dtype=torch.bool);legal[:,2]=False
    assert torch.equal(head(x,legal)[0],base(x,legal)[0])
    saved={k:v.clone() for k,v in head.base.state_dict().items()}
    opt=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=.01)
    for _ in range(3):
        opt.zero_grad();loss=-head(x,legal)[0].log_softmax(-1)[:,0].mean();loss.backward();opt.step()
    assert head.keys.weight.grad.abs().sum()>0
    assert all(torch.equal(v,head.base.state_dict()[k]) for k,v in saved.items())
    assert (head(x,legal)[0].softmax(-1)[:,2]==0).all()
    _,weights=head.context(x)
    assert weights.shape==(6,4,32) and torch.allclose(weights.sum(-1),torch.ones(6,4))
    changed=x.clone();changed[:,5:]+=torch.randn_like(changed[:,5:])*2
    assert not torch.allclose(head(changed,legal)[0][:,0],head(x,legal)[0][:,0])

def test_reject_summary_only_pack():
    head=ContextResidualHead(DecisionValueHead(dim=8,width=6))
    with pytest.raises(ValueError):head.context(torch.zeros(2,5,8))
