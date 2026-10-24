import copy
import numpy as np
import pytest
from lsdrag.decision_step_credit import generated_by_step,token_returns,group_step_advantages

def episode(tokens,q=.5,forced=None):
    ds=[dict(forced=i in (forced or []),branch_intervention=False) for i in range(len(tokens))]
    calls=[]
    for n in tokens:calls.extend([dict(pack_slots=37,generated_tokens=0),dict(generated_tokens=n)])
    return dict(qid='trainq',decisions=ds,reader_calls=calls,generated_tokens=sum(tokens),f1=q)

def test_future_forced_cost_included_and_past_excluded():
    e=episode([10,20,30],forced=[2]);r=token_returns(e,coefficient=.01)
    assert np.allclose(r,[-.1,0,.2])
    changed=episode([100,20,30],forced=[2]);assert token_returns(changed,.01)[1]==r[1]
    adv=group_step_advantages([e,episode([10,20,30],forced=[2])],scale=.1)
    assert all(len(a)==2 for a in adv)

def test_same_first_advantage_and_fixed_scale_equal_reward_group():
    es=[episode([10,20]),episode([20,10])]
    total=group_step_advantages(es,.1,remaining=False);remaining=group_step_advantages(es,.1)
    assert all(a[0]==b[0] for a,b in zip(total,remaining))
    assert all(np.isfinite(a).all() and np.abs(a).max()<1 for a in remaining)
    assert np.allclose(total,0)

def test_zero_quality_gate():
    assert all(not np.any(a) for a in group_step_advantages([episode([1],0),episode([100],0)],.1))

def test_accounting_and_intervention_rejected():
    e=episode([10]);e['generated_tokens']=11
    with pytest.raises(ValueError):generated_by_step(e)
    e=episode([10]);e['decisions'][0]['branch_intervention']=True
    with pytest.raises(ValueError):generated_by_step(e)

def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):group_step_advantages([episode([1]),episode([2])],0)
    with pytest.raises(ValueError):token_returns(episode([-1]))
