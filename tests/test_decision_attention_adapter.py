import pytest
import torch
from lsdrag.decision_state_learning import StateHead
from lsdrag.decision_attention_adapter import AttentionAdapter


@pytest.mark.parametrize('variant', ['question_attention', 'action_attention', 'action_set'])
def test_identity_learned_context_and_legal_mask(variant):
    torch.manual_seed(41)
    x = torch.randn(6, 37, 8)
    mask = torch.ones(6, 4, dtype=torch.bool)
    mask[:, 2] = False
    base = StateHead(dim=8, inputs='multi')
    base.normalize_from_training(x)
    head = AttentionAdapter(base, variant, width=8)
    assert torch.equal(base(x, mask)[0], head(x, mask)[0])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
    for _ in range(4):
        optimizer.zero_grad()
        loss = -head(x, mask)[0].log_softmax(-1)[:, 1].mean()
        loss.backward()
        optimizer.step()
    changed = x.clone()
    changed[:, 5:37] = torch.randn_like(changed[:, 5:37])
    assert not torch.equal(head(x, mask)[0], head(changed, mask)[0])
    assert head.key.weight.grad.abs().sum() > 0
    assert (head(x, mask)[0].softmax(-1)[:, 2] == 0).all()
    assert all(torch.isfinite(p).all() for p in head.parameters())
