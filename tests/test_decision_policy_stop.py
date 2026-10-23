import torch
from lsdrag.decision_policy import PolicyHead
from lsdrag.decision_policy_stop import StopOnlyHead
from lsdrag.decision_policy_followup import fit_improvement

def fixture():
    torch.manual_seed(41);d=7
    hist={'action_weight':torch.randn(4,d),'action_bias':torch.randn(4)}
    base=PolicyHead(d,'mlp');h=StopOnlyHead(hist,base)
    x=torch.randn(12,37,d);legal=torch.ones(12,4,dtype=torch.bool);legal[::2,2]=False
    return h,x,legal,hist

def test_initial_greedy_matches_historical_bc():
    h,x,l,hist=fixture();expected=(x[:,0]@hist['action_weight'].T+hist['action_bias']).masked_fill(~l,-1e9).argmax(-1)
    assert torch.equal(h(x,l)[0].argmax(-1),expected)

def test_learning_can_change_stop_but_never_tool_choice():
    h,x,l,hist=fixture();initial=h.tool_weight.clone();q=torch.full((12,4),-1.);q[:,3]=1
    fitted,_=fit_improvement(h,h,x,l,q,[str(i) for i in range(12)],'return_pg',epochs=15)
    logits=fitted(x,l)[0];expected=(x[:,0]@hist['action_weight'].T+hist['action_bias'])[:,:3].masked_fill(~l[:,:3],-1e9).argmax(-1)
    assert torch.equal(logits[:,:3].argmax(-1),expected)
    assert torch.equal(fitted.tool_weight,initial)
    assert fitted(x,l)[0].softmax(-1)[:,3].mean()>h(x,l)[0].softmax(-1)[:,3].mean()

def test_forced_final_and_illegal_mask():
    h,x,l,hist=fixture();l[:]=False;l[:,3]=True
    p=h(x,l)[0].softmax(-1)
    assert torch.equal(p[:,3],torch.ones(12))
    assert torch.equal(p[:,:3],torch.zeros(12,3))
