import torch
from lsdrag.decision_safe_stop_targets import safe_stop_targets

def row(qid,ref,alt):
    return dict(qid=qid,turn=0,reference_probs=[.8,0,0,.2],legal=torch.tensor([True,False,False,True]),samples=[dict(action=0,f1=ref[0],em=ref[1],generated=ref[2],errors=[]),dict(action=3,f1=alt[0],em=alt[1],generated=alt[2],errors=[])])

def test_em_loss_cannot_buy_cost_saving():
    labels,_=safe_stop_targets([row('q',(1,1,100),(1,0,1))])
    assert labels=={}

def test_correct_preserving_saving_selected():
    labels,_=safe_stop_targets([row('q',(1,1,100),(1,1,10))])
    assert labels=={0:3}

def test_zero_zero_saving_not_positive_target():
    labels,_=safe_stop_targets([row('q',(0,0,100),(0,0,1))])
    assert labels=={}
