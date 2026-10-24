"""Native shared-corpus evaluator with one anchored post-reader policy type."""
import torch
from lsdrag.decision_shared_residual import SharedResidualPolicy
from lsdrag.decision_state_learning import StateHead
from lsdrag.decision_stop_value import StopValueHead
from evaluate_decision_stop_value_h2 import load_head as original
import evaluate_decision_shared_corpus as evaluator

def load_head(spec,device='cpu'):
    if spec['kind']!='shared_residual':return original(spec,device)
    head=SharedResidualPolicy(StopValueHead(StateHead(inputs='multi',body='mlp')))
    head.load_state_dict(torch.load(spec['checkpoint'],map_location='cpu',weights_only=True))
    return head.to(device).eval()
if __name__=='__main__':
    evaluator.load_head=load_head;evaluator.main()
