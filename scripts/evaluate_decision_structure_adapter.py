import torch
from lsdrag.decision_state_learning import StateHead
from lsdrag.decision_structure_adapter import StructureAdapter
from evaluate_decision_value_heads import load_head as original
import evaluate_decision_shared_corpus as evaluator

def load_head(spec,device='cpu'):
    if spec['kind']!='structure_adapter':return original(spec,device)
    h=StructureAdapter(StateHead(inputs='multi',body='mlp'),spec['variant'])
    h.load_state_dict(torch.load(spec['checkpoint'],map_location='cpu',weights_only=True))
    return h.to(device).eval()
if __name__=='__main__':
    evaluator.load_head=load_head;evaluator.main()
