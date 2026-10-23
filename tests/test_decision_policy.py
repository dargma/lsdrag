import torch
from lsdrag.decision_policy import PolicyHead, distribution, returns_for_episode, clipped_policy_loss, pack_hidden


def test_legality_and_sampling_logprob_replay():
    logits=torch.tensor([[20.,1.,99.,0.]],requires_grad=True)
    legal=torch.tensor([[True,True,False,True]])
    dist=distribution(logits,legal,.05)
    assert dist.probs[0,2]==0
    torch.manual_seed(9);actions=dist.sample((100,))
    assert not (actions==2).any()
    replay=distribution(logits.clone(),legal,.05)
    torch.testing.assert_close(dist.log_prob(actions),replay.log_prob(actions))
    assert torch.isfinite(dist.entropy()).all()


def test_cost_to_go_not_full_episode_cost_repeated():
    steps=[{'cost_units':2},{'cost_units':3},{'cost_units':1}]
    assert returns_for_episode(steps,1.,.1)==[.39999999999999997,.6,.9]
    assert returns_for_episode(steps,0.,0.)==[0.,0.,0.]


def test_advantage_updates_policy_in_correct_directions():
    logp=torch.tensor([-.5,-.5],requires_grad=True)
    clipped_policy_loss(logp,logp.detach(),torch.tensor([1.,-1.])).backward()
    assert logp.grad[0]<0 and logp.grad[1]>0


def test_span_features_do_not_read_future_position():
    h=torch.randn(64,8);first=pack_hidden(h,(8,12),(32,48))
    h[48:]+=100;second=pack_hidden(h,(8,12),(32,48))
    torch.testing.assert_close(first[1:3],second[1:3])
    torch.testing.assert_close(first[4],second[4])
    assert not torch.allclose(first[0],second[0])


def test_auxiliary_training_cannot_change_actor_if_trunk_frozen():
    torch.manual_seed(2);h=PolicyHead(8,'mlp');x=torch.randn(3,37,8);legal=torch.ones(3,4,dtype=torch.bool)
    before=h(x,legal)[0].detach().clone()
    for p in h.parameters():p.requires_grad_(False)
    for p in h.success.parameters():p.requires_grad_(True)
    opt=torch.optim.SGD(h.success.parameters(),lr=.1)
    opt.zero_grad();h(x,legal)[2].sum().backward();opt.step()
    torch.testing.assert_close(before,h(x,legal)[0])


def test_all_readouts_output_finite_policy_and_scalar_value():
    for arch in ['linear','mlp','attention']:
        h=PolicyHead(8,arch);x=torch.randn(4,37,8);mask=torch.ones(4,4,dtype=torch.bool);mask[:,2]=False
        logits,value,success,cost=h(x,mask)
        assert value.shape==(4,) and success.shape==cost.shape==(4,4)
        distribution(logits,mask).log_prob(torch.zeros(4,dtype=torch.long)).sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in h.parameters() if p.grad is not None)
