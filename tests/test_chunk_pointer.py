import pytest
import torch

from lsdrag.chunk_pointer import ChunkPointer, pointer_target, pool_candidates, unread_candidates


@pytest.mark.parametrize('k', [1, 5, 8, 17])
def test_dynamic_candidates_and_exact_teacher_replay(k):
    torch.manual_seed(11)
    head = ChunkPointer(12, 4)
    state, candidates = torch.randn(12), torch.randn(k, 12)
    selected, sampled = head.select(state, candidates, stochastic=True)
    replayed, replay = head.select(state, candidates, targets=selected)
    assert replayed == selected
    assert 1 <= len(selected) <= min(8, k)
    assert len(selected) == len(set(selected))
    assert len(sampled) == len(replay)
    assert torch.equal(torch.stack([x['logp'] for x in sampled]), torch.stack([x['logp'] for x in replay]))
    assert not sampled[0]['stop']
    if len(selected) == min(8, k):
        assert not sampled[-1]['stop']
    else:
        assert sampled[-1]['stop']


def test_gradients_to_state_and_visible_candidate_tokens():
    head = ChunkPointer(12, 4)
    hidden = torch.randn(20, 12, requires_grad=True)
    candidates = pool_candidates(hidden, [(1, 4), (6, 9), (11, 14)])
    _, events = head.select(hidden[-1], candidates, targets=[1])
    loss = -torch.stack([e['logp'] for e in events]).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert hidden.grad[-1].abs().sum() > 0
    assert hidden.grad[1:4].abs().sum() > 0
    assert hidden.grad[15:19].abs().sum() == 0
    for p in head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_no_first_stop_duplicate_or_unobserved_target():
    head = ChunkPointer(12, 4)
    h, c = torch.randn(12), torch.randn(5, 12)
    assert torch.isneginf(head.logits(h, c)[-1])
    assert torch.isneginf(head.logits(h, c, [2])[2])
    for target in ([], [1, 1], [5]):
        with pytest.raises(ValueError):
            head.select(h, c, targets=target)
    with pytest.raises(ValueError):
        head.select(h, c[:0])


def test_unread_and_offline_label_provenance():
    rows = [{'chunk_id': x, 'title': x, 'snippet': x} for x in ['b', 'a', 'c']]
    history = [{'tool_name': 'semantic_search', 'arguments': {}, 'result': rows},
               {'tool_name': 'read_chunk', 'arguments': {'chunk_ids': ['b']}, 'result': []}]
    ids = [x[0] for x in unread_candidates(history)]
    assert ids == ['a', 'c']
    assert pointer_target(ids, ['a'], {'c'}) == ([1], 'gold_present')
    assert pointer_target(ids, ['a'], {'z'}) == ([0], 'teacher_fallback')
    assert pointer_target(ids, ['a', 'a'], {'z'}) == (None, 'invalid_teacher')
