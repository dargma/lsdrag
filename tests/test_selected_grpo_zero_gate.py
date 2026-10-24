import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_decision_selected_head_grpo_zero_gate import quality_gated_advantages
from lsdrag.decision_grpo import group_advantages

def test_zero_quality_removes_cheap_failure_preference():
    rewards=[-.03,-.09,-.2,-.4]
    assert group_advantages(rewards)[0]>0
    assert np.array_equal(quality_gated_advantages(rewards,[0]*4),np.zeros(4))

def test_positive_quality_preserves_original_update():
    rewards=[.2,-.09,.5,-.4]
    assert np.array_equal(quality_gated_advantages(rewards,[.3,0,.8,0]),group_advantages(rewards))

@pytest.mark.parametrize('f1s',[[0,0],[0,0,float('nan'),0],[0,0,-1,0]])
def test_reject_invalid_quality(f1s):
    with pytest.raises(ValueError):quality_gated_advantages([0]*4,f1s)
