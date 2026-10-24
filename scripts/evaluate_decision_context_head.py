"""Shared evaluation with an optional context-residual checkpoint loader."""
import torch
from lsdrag.decision_value_head import DecisionValueHead
from lsdrag.decision_context_residual import ContextResidualHead
from evaluate_decision_value_heads import load_head as value_load_head
import evaluate_decision_shared_corpus as evaluator

def load_head(spec,device='cpu'):
    if spec['kind']!='context_residual':return value_load_head(spec,device)
    head=ContextResidualHead(DecisionValueHead(body='mlp'),width=spec['width'])
    head.load_state_dict(torch.load(spec['checkpoint'],map_location='cpu',weights_only=True))
    return head.to(device).eval()
if __name__=='__main__':
    evaluator.load_head=load_head
    evaluator.main()
