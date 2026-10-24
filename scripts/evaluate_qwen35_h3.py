import torch
from lsdrag.decision_state_learning import StateHead
from lsdrag.decision_stop_value import StopValueHead
from lsdrag.decision_qwen35_reader import Qwen35Reader
import evaluate_decision_shared_corpus as evaluator

def load_head(spec,device='cpu'):
    head=StateHead(dim=4096,inputs='multi',body='mlp')
    if spec['kind']=='qwen35_stop':head=StopValueHead(head)
    else:assert spec['kind']=='qwen35_base'
    head.load_state_dict(torch.load(spec['checkpoint'],weights_only=True,map_location='cpu'));return head.to(device).eval()
if __name__=='__main__':
    evaluator.RecoveryPolicyReader=Qwen35Reader;evaluator.load_head=load_head;evaluator.main()
