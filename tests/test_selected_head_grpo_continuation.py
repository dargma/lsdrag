import hashlib
import importlib.util
import sys
from pathlib import Path

import torch
import numpy as np
import time
from lsdrag.decision_policy import rollout
from lsdrag.decision_policy import distribution
from lsdrag.decision_state_learning import StateHead


def test_selected_checkpoint_is_exact_initialization_and_updates_without_norm_drift(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('selected_grpo', scripts / 'run_decision_selected_head_grpo_r1.py')
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


def test_rollout_fp32_storage_replays_actual_behavior_without_quantization():
    torch.manual_seed(17)
    pack = torch.randn(37, 2560) / 7
    head = StateHead(inputs='multi', body='mlp')

    class Reader:
        pending = None

        def capture_pack(self, messages, history, deadline):
            return pack, {'prompt_tokens': 8, 'generated_tokens': 0}

        def continue_cached(self, name, history, deadline):
            assert name == 'FINAL'
            return {'answer': 'x', 'support_ids': []}, {'prompt_tokens': 0, 'generated_tokens': 1}, None

    args = (Reader(), 'question', [], np.zeros((0, 768)), None, 'prompt', head, time.perf_counter()+30)
    _, rows, _ = rollout(*args, first_action=3, state_storage_dtype=torch.float32)
    row = rows[0]
    assert row['pack'].dtype == torch.float32 and torch.equal(row['pack'], pack)
    legal = row['legal'][None]
    replay = distribution(head(row['pack'][None], legal)[0], legal).log_prob(torch.tensor([3]))
    assert torch.allclose(replay, torch.tensor([row['old_logp']]), atol=1e-7, rtol=1e-7)
    _, legacy_rows, _ = rollout(*args, first_action=3)
    assert legacy_rows[0]['pack'].dtype == torch.bfloat16
    assert not torch.equal(legacy_rows[0]['pack'].float(), pack)
