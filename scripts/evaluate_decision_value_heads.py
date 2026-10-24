"""Use unchanged shared-corpus evaluator with one additional checkpoint type."""
import torch
from lsdrag.decision_value_head import DecisionValueHead
import evaluate_decision_shared_corpus as evaluator
original_load_head=evaluator.load_head

def load_head(spec,device='cpu'):
    if spec['kind']!='decision_value':return original_load_head(spec,device)
    head=DecisionValueHead(body=spec['body'])
    head.load_state_dict(torch.load(spec['checkpoint'],weights_only=True,map_location='cpu'))
    return head.to(device).eval()

if __name__=='__main__':
    evaluator.load_head=load_head
    evaluator.main()
