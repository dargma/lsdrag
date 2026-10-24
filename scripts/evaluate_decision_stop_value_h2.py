"""Reuse the native-contract evaluator with the frozen-base stopping head."""
import torch
from lsdrag.decision_state_learning import StateHead
from lsdrag.decision_stop_value import StopValueHead
from evaluate_decision_attention import load_head as original
import evaluate_decision_shared_corpus as evaluator


def load_head(spec, device='cpu'):
    if spec['kind'] != 'stop_value_h2':
        return original(spec, device)
    head = StopValueHead(StateHead(inputs='multi', body='mlp'))
    head.load_state_dict(torch.load(spec['checkpoint'], weights_only=True, map_location='cpu'))
    return head.to(device).eval()


if __name__ == '__main__':
    evaluator.load_head = load_head
    evaluator.main()
