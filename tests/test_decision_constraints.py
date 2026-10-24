import torch
from lsdrag.decision_constraints import constrained_loss,feasibility

def test_lp_known_quality_budget_bound_and_illegal_action_exclusion():
    q=torch.tensor([[1.,0.,99.,99.]])
    g=torch.tensor([[10.,0.,0.,0.]])
    legal=torch.tensor([[True,True,False,False]])
    result=feasibility(q,g,legal,torch.ones(1),5.,.6)
    assert result['solver_success'] and abs(result['quality_upper_bound']-.5)<1e-7
    assert not result['quality_floor_feasible']

def test_token_dual_pushes_toward_cheaper_action_and_quality_dual_reverses():
    q=torch.tensor([[1.,0.]])
    g=torch.tensor([[10.,0.]])
    legal=torch.ones(1,2,dtype=torch.bool)
    for dq,expensive_decreases in [(0.,True),(10.,False)]:
        logits=torch.zeros(1,2,requires_grad=True)
        loss,*_=constrained_loss(logits,legal,q,g,torch.ones(1),torch.ones(1,2)/2,2.,.8,2.,dq)
        loss.backward()
        assert (logits.grad[0,0]>0).item()==expensive_decreases
