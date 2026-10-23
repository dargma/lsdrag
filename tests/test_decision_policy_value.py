import pytest
import torch
from lsdrag.decision_policy_value import RidgeValueHead,fit_ridge

def test_exact_weighted_ridge_matches_primal_and_masks_illegal():
    torch.manual_seed(3);x=torch.randn(9,37,5);l=torch.ones(9,4,dtype=torch.bool);l[:2,2]=False;y=torch.randn(9,4)
    h=RidgeValueHead(torch.zeros(5),torch.ones(5));h,stats=fit_ridge(h,x,l,y,[str(i) for i in range(9)],penalty=.7)
    for a in range(4):
        z=x[l[:,a],0].double();v=y[l[:,a],a].double();xm=z.mean(0);ym=v.mean();z=z-xm;v=v-ym
        w=torch.linalg.solve(z.T@z/len(z)+.7*torch.eye(5,dtype=torch.double),z.T@v/len(z))
        assert torch.allclose(h.weight[a].double(),w,atol=1e-6)
    assert (h(x,l)[0][~l]==-1e9).all()

def test_missing_action_coverage_rejected():
    h=RidgeValueHead(torch.zeros(3),torch.ones(3));l=torch.ones(4,4,dtype=torch.bool);l[:,2]=False
    with pytest.raises(ValueError,match='coverage'):fit_ridge(h,torch.zeros(4,37,3),l,torch.zeros(4,4),['a']*4)
