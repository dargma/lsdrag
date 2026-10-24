import torch
from lsdrag.decision_state_learning import StateHead
from lsdrag.decision_stop_value import StopValueHead
from lsdrag.decision_progress_stop import ProgressStopPolicy,progress_features

def test_progress_is_causal_and_initial_zero():
    z=torch.tensor([[1.,3.]]);prev=torch.tensor([[.5,1.]])
    assert torch.equal(progress_features(z,None,None,0),torch.zeros(1,6))
    assert torch.equal(progress_features(z,prev,1,2),torch.tensor([[.5,2.,0.,1.,0.,.25]]))

def test_episode_reset_prevents_cross_question_state_and_preserves_initial_h3():
    torch.set_num_threads(2);torch.manual_seed(20261085)
    anchor=StopValueHead(StateHead(inputs='multi',body='mlp')).eval();head=ProgressStopPolicy(anchor).eval()
    x=torch.randn(1,37,2560);legal=torch.ones(1,4,dtype=torch.bool)
    with torch.no_grad():
        # Make the first action nonterminal so a second causal state exists.
        anchor.stop.bias.fill_(-100);head.anchor.stop.bias.fill_(-100)
        head.progress.weight.fill_(.1)
        ref=anchor(x,legal)[0];first=head(x,legal)[0]
        assert torch.equal(ref,first)
        second=head(x,legal)[0];assert torch.equal(ref[:,:3],second[:,:3])
        assert not torch.equal(first[:,3],second[:,3])
        head.reset_episode();assert torch.equal(first,head(x,legal)[0])
        assert all('previous' not in k and 'turn' not in k for k in head.state_dict())
