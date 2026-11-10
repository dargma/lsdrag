import pytest

from lsdrag.fullft import v22_terminal_reward, v22_step_rewards, v22_group_advantages


def ep(em=1, f1=1., calls=3, tokens=128):
    return {'qid': 'q', 'em': em, 'f1': f1, 'answer': 'answer', 'errors': [],
            'tool_calls': calls, 'generated_tokens': tokens,
            'decisions': [{}]*(calls+1), 'trajectory': []}


def test_correctness_ordering_and_efficiency_monotonicity():
    worst_correct = v22_terminal_reward(ep(calls=8, tokens=10**9))['terminal']
    best_incorrect = v22_terminal_reward(ep(0, 1., calls=0, tokens=0))['terminal']
    assert worst_correct >= .75 > best_incorrect + .10
    for n in range(8):
        assert v22_terminal_reward(ep(calls=n))['terminal'] > v22_terminal_reward(ep(calls=n+1))['terminal']
    for em, f1 in [(1, 1.), (0, .5)]:
        assert v22_terminal_reward(ep(em, f1, tokens=50))['terminal'] > v22_terminal_reward(ep(em, f1, tokens=200))['terminal']
    assert v22_terminal_reward(ep(0, 0., calls=0, tokens=0))['terminal'] == 0
    failed = ep(); failed['errors'] = ['bad JSON']
    assert v22_terminal_reward(failed)['terminal'] == 0


def test_process_credit_once_and_bounded():
    e = ep(calls=4)
    search = {'tool_name': 'semantic_search', 'arguments': {}, 'result': [{'chunk_id': 'a'}]}
    read = {'tool_name': 'read_chunk', 'arguments': {'chunk_ids': ['a']}, 'result': []}
    e['trajectory'] = [search, search, read, read]
    r = v22_step_rewards(e, {'a'}, process=True)
    assert r[:4] == [.05, 0., .05, 0.]
    assert sum(r) == pytest.approx(v22_terminal_reward(e)['terminal']+.1)


def test_signed_step_advantage_and_ended_trajectory_baseline():
    group = [ep(0, 0., calls=0) for _ in range(7)] + [ep(1, 1., calls=1)]
    a = v22_group_advantages(group, {'a'}, process=True)
    reward = v22_terminal_reward(group[-1])['terminal']
    assert a[0] == pytest.approx([-reward/7])
    assert a[-1] == pytest.approx([reward, reward])
    scalar = v22_group_advantages(group, {'a'}, process=False)
    assert scalar[0][0] == pytest.approx(-reward/8)
    assert scalar[-1] == pytest.approx([reward*7/8]*2)
    with pytest.raises(ValueError):
        v22_group_advantages(group[:7], {'a'})
