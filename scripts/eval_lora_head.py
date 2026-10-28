"""Run evaluate_decision_shared_corpus.py with LoRA weights attached to the 4B reader.

Usage: LORA_PATH=<lora_iterationN.pt> python scripts/eval_lora_head.py --spec <spec.json>
Decoding stays greedy; only the reader weights differ from the plain evaluator.
"""
import os
from lsdrag.decision_policy_recovery import RecoveryPolicyReader
from lsdrag.lora4b import attach_lora, load_lora
import evaluate_decision_shared_corpus as evaluator


class LoraReader(RecoveryPolicyReader):
    def __init__(self, path, config):
        super().__init__(path, config)
        state = os.environ['LORA_PATH']
        rank = next(v.shape[0] for k, v in __import__('torch').load(state, weights_only=True).items() if k.endswith('lora_A'))
        attach_lora(self.model, rank, 2*rank)
        load_lora(self.model, state)
        self.model.requires_grad_(False)


evaluator.RecoveryPolicyReader = LoraReader

if __name__ == '__main__':
    evaluator.main()
