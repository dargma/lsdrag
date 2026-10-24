import torch
from lsdrag.decision_state_learning import StateHead
from lsdrag.decision_stop_value import StopValueHead
from lsdrag.decision_tool_return import ToolReturnPolicy
from evaluate_decision_stop_value_h2 import load_head as original
import evaluate_decision_shared_corpus as evaluator

def load_head(spec,device='cpu'):
    if spec['kind']!='tool_return':return original(spec,device)
    head=ToolReturnPolicy(StopValueHead(StateHead(inputs='multi',body='mlp')),constant=spec['constant'])
    head.load_state_dict(torch.load(spec['checkpoint'],weights_only=True,map_location='cpu'))
    return head.to(device).eval()

if __name__=='__main__':
    evaluator.load_head=load_head
    evaluator.main()
