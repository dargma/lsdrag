import numpy as np
from lsdrag.decision_constrained_stop import fit_constrained_stop

def test_shared_feature_cannot_flip_protected_action():
    x=np.array([[1.],[1.]])
    delta,e=fit_constrained_stop(x,[-.1,-.1],[1.,.475],[.5,.5],[1],[-.1,-.1],[.5,.5])
    assert e['success'] and e['constraint_violation']<1e-7
    assert -.1+delta[0]<=-.01+1e-7

def test_separable_state_can_change_without_protected_change():
    x=np.eye(2)
    delta,e=fit_constrained_stop(x,[-.01,-.1],[1.,.475],[.5,.5],[1],[-.01,-.1],[.5,.5])
    assert e['success'] and e['kl']<=.01+1e-7
    assert -.01+delta[0]>0
    assert -.1+delta[1]<0
