import io
import torch

from lsdrag.fullft import MasterAdamW


def test_master_optimizer_restores_fp32_values_moments_and_next_update():
    p = torch.nn.Parameter(torch.tensor([.1, .2, .3], dtype=torch.bfloat16))
    original = MasterAdamW([p], lr=.01)
    p.grad = torch.tensor([.25, -.3, .5], dtype=p.dtype)
    original.step(); original.zero_grad()
    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer); buffer.seek(0)
    q = torch.nn.Parameter(torch.zeros_like(p))
    restored = MasterAdamW([q], lr=.01)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(p, q)
    assert torch.equal(original.master[id(p)], restored.master[id(q)])
    for param, opt in ((p, original), (q, restored)):
        param.grad = torch.tensor([.7, .1, -.2], dtype=param.dtype)
        opt.step(); opt.zero_grad()
    assert torch.equal(p, q)
    assert torch.equal(original.master[id(p)], restored.master[id(q)])
