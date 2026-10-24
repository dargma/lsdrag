import torch
from lsdrag.decision_state_learning import StateHead
from lsdrag.decision_stop_value import StopValueHead, stop_margin, supported_stop_label


def test_exact_initialization_and_only_final_changes_after_learning():
    torch.manual_seed(3)
    pack = torch.randn(5, 37, 8)
    legal = torch.ones(5, 4, dtype=torch.bool)
    legal[0, 2] = False
    base = StateHead(dim=8, inputs='multi', body='mlp')
    head = StopValueHead(base)
    assert torch.equal(head(pack, legal)[0], base(pack, legal)[0])
    before = {k: v.clone() for k, v in head.base.state_dict().items()}
    opt = torch.optim.AdamW(head.stop.parameters(), lr=.01)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        stop_margin(head, pack, legal), torch.ones(5))
    loss.backward()
    opt.step()
    assert torch.equal(head(pack, legal)[0][:, :3], base(pack, legal)[0][:, :3])
    assert not torch.equal(head(pack, legal)[0][:, 3], base(pack, legal)[0][:, 3])
    assert all(torch.equal(v, before[k]) for k, v in head.base.state_dict().items())
    assert all(p.grad is None for p in head.base.parameters())


def record(stop_f1, cont_f1, stop_tokens=10, cont_tokens=100):
    return dict(continue_action=1, samples=[
        dict(action=3, f1=stop_f1, generated=stop_tokens, errors=[]),
        dict(action=1, f1=cont_f1, generated=cont_tokens, errors=[])])


def test_never_teach_cheap_failure_or_errored_comparison():
    assert supported_stop_label(record(0, 0)) is None
    assert supported_stop_label(record(1, 1)) == 1
    assert supported_stop_label(record(0, 1)) == 0
    assert supported_stop_label(record(1, 0)) == 1
    assert supported_stop_label(record(.5, .5, 100, 10)) == 0
    bad = record(1, 0)
    bad['samples'][0]['errors'] = ['invalid format']
    assert supported_stop_label(bad) is None


def test_illegal_final_stays_masked_even_with_large_bias():
    head = StopValueHead(StateHead(dim=8, inputs='multi', body='mlp'))
    with torch.no_grad():
        head.stop.bias.fill_(1000)
    pack = torch.zeros(1, 37, 8)
    legal = torch.tensor([[True, True, False, False]])
    assert head(pack, legal)[0][0, 3] == -1e9
