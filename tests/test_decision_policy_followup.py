import torch
from lsdrag.decision_policy import PolicyHead,distribution
from lsdrag.decision_policy_followup import tempered_initialization,state_snapshots,state_weights,improvement_loss


def test_temperature_preserves_greedy_and_increases_entropy():
    for architecture in ['linear','mlp']:
        torch.manual_seed(4);h=PolicyHead(8,architecture);x=torch.randn(4,37,8);mask=torch.ones(4,4,dtype=torch.bool);mask[:,2]=False
        a=distribution(h(x,mask)[0],mask);b=distribution(tempered_initialization(h,8)(x,mask)[0],mask)
        assert torch.equal(a.probs.argmax(-1),b.probs.argmax(-1))
        assert (b.entropy()>=a.entropy()).all()


def test_prefix_snapshots_include_late_and_exclude_future_results():
    ts=[{'tool_name':'semantic_search','arguments':{'query':str(i)},'result':[{'chunk_id':f'p{i}'}]} for i in range(8)]
    snaps=state_snapshots('question','prompt',ts)
    assert [s['turn'] for s in snaps]==[0,3,7]
    for s in snaps:
        assert len(s['history'])==s['turn']
        assert f'p{s["turn"]}' not in str(s['messages'])
    assert [s['turn'] for s in state_snapshots('q','p',ts[:2])]==[0,1,2]


def test_q_gradient_favors_best_action_and_never_illegal():
    z=torch.zeros(1,4,requires_grad=True);mask=torch.tensor([[True,True,False,True]])
    q=torch.tensor([[0.,.1,999.,1.]])
    loss,_=improvement_loss(z,mask,q,distribution(z,mask).probs.detach(),torch.ones(1),'return_pg',kl_coef=0)
    loss.backward();assert z.grad[0,3]<0 and z.grad[0,0]>0 and z.grad[0,2]==0


def test_long_episodes_do_not_get_more_question_weight():
    w=state_weights(['a','a','a','b'],'cpu')
    torch.testing.assert_close(w[:3].sum(),w[3])


def test_same_quality_cost_prefers_final():
    z=torch.zeros(1,4,requires_grad=True);mask=torch.tensor([[True,True,False,True]])
    quality=torch.ones(1,4);cost=torch.tensor([[4.,3.,0.,.5]])
    q=quality-.03*cost
    loss,_=improvement_loss(z,mask,q,distribution(z,mask).probs.detach(),torch.ones(1),'return_pg',kl_coef=0)
    loss.backward();assert z.grad[0,3]<z.grad[0,0]
