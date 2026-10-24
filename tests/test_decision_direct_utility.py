import torch
from lsdrag.decision_state_learning import StateHead
from lsdrag.decision_direct_utility import DirectUtilityHead


def test_predictions_determine_selection_and_receive_gradients():
    torch.manual_seed(7)
    head = DirectUtilityHead(StateHead(dim=8, inputs='multi'))
    x = torch.randn(6, 37, 8)
    legal = torch.ones(6, 4, dtype=torch.bool)
    legal[:, 2] = False
    logits, _, quality, cost = head(x, legal)
    expected = ((quality - .03 * cost) / .1).masked_fill(~legal, -1e9)
    assert torch.equal(logits, expected)
    assert ((quality > 0) & (quality < 1)).all() and (cost > 0).all()
    assert (logits.softmax(-1)[:, 2] == 0).all()
    ((quality - .8).square()[legal].mean() + (cost - 2).square()[legal].mean()).backward()
    assert head.encoder.weight.grad.abs().sum() > 0
    assert head.actions.weight.grad.abs().sum() > 0
    assert (head.actions.weight.grad[2] == 0).all()
