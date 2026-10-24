import hashlib
import importlib.util
import sys
from pathlib import Path

import torch
from lsdrag.decision_policy import distribution
from lsdrag.decision_state_learning import StateHead


def test_selected_checkpoint_is_exact_initialization_and_updates_without_norm_drift(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('selected_grpo', scripts / 'run_decision_selected_head_grpo.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    torch.manual_seed(5)
    source = StateHead(inputs='multi', body='mlp')
    x = torch.randn(4, 5, 2560)
    source.normalize_from_training(x)
    checkpoint = tmp_path / 'outcome.pt'
    torch.save(source.state_dict(), checkpoint)
    descriptor = dict(kind='structure', inputs='multi', body='mlp', checkpoint=str(checkpoint),
                      sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    anchor, learner = module.continuation_heads(descriptor)
    assert all(torch.equal(v, learner.state_dict()[k]) for k, v in source.state_dict().items())
    assert all(not p.requires_grad for p in anchor.parameters())
    assert all(p.requires_grad for p in learner.parameters())
    legal = torch.ones(4, 4, dtype=torch.bool)
    legal[:, 2] = False
    action = torch.zeros(4, dtype=torch.long)
    old = distribution(anchor(x, legal)[0], legal).log_prob(action).detach()
    before = distribution(learner(x, legal)[0], legal).probs.detach()
    opt = torch.optim.AdamW(learner.parameters(), lr=1e-4)
    ratio = (distribution(learner(x, legal)[0], legal).log_prob(action) - old).exp()
    (-torch.minimum(ratio, ratio.clamp(.8, 1.2))).mean().backward()
    opt.step()
    after = distribution(learner(x, legal)[0], legal).probs.detach()
    assert after[:, 0].mean() > before[:, 0].mean()
    assert (after[:, 2] == 0).all()
    assert torch.equal(learner.mean, source.mean) and torch.equal(learner.scale, source.scale)
    assert all(torch.equal(v, anchor.state_dict()[k]) for k, v in source.state_dict().items())
