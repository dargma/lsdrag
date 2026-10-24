import torch
from lsdrag.decision_state_learning import StateHead
from lsdrag.decision_attention_adapter import AttentionAdapter
from evaluate_decision_structure_adapter import load_head as original
import evaluate_decision_shared_corpus as evaluator


def load_head(spec, device='cpu'):
    if spec['kind'] != 'attention_adapter':
        return original(spec, device)
    head = AttentionAdapter(StateHead(inputs='multi', body='mlp'), spec['variant'])
    head.load_state_dict(torch.load(spec['checkpoint'], map_location='cpu', weights_only=True))
    return head.to(device).eval()


if __name__ == '__main__':
    evaluator.load_head = load_head
    evaluator.main()
