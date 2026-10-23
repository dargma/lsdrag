import numpy as np
import pytest
from lsdrag.decision_grpo import group_advantages

def test_equal_group_no_advantage():
 assert np.array_equal(group_advantages([.3]*4),np.zeros(4))

def test_advantage_order_and_center():
 a=group_advantages([0,.2,.6,1.]);assert np.all(np.diff(a)>0);assert abs(a.mean())<1e-8
 assert np.allclose(a,group_advantages([3,3.2,3.6,4]))

def test_bad_group_rejected():
 with pytest.raises(ValueError):group_advantages([0,float('nan')])
