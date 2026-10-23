import torch
from lsdrag.decision_rule import SemanticOnceHead

def test_rule_does_not_depend_on_observed_read_legality_and_resets():
 h=SemanticOnceHead();x=torch.zeros(1,37,8);legal=torch.tensor([[True,True,False,True]])
 assert h(x,legal)[0].argmax().item()==1
 # Empty search result must still terminate rather than repeat search.
 assert h(x+10,legal)[0].argmax().item()==3
 h.reset_episode();assert h(x,legal)[0].argmax().item()==1
