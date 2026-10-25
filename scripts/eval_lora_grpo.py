"""Greedy evaluation through the Qwen3.5 evaluator path, optionally with a trained LoRA (env LORA_PATH)."""
import os
from lsdrag.decision_qwen35_reader import Qwen35Reader
from lsdrag.lora_grpo import attach_lora, load_lora
import evaluate_decision_shared_corpus as evaluator
from evaluate_qwen35_h3 import load_head


class LoraReader(Qwen35Reader):
    def __init__(self, path, config):
        super().__init__(path, config)
        if os.environ.get('LORA_PATH'):
            attach_lora(self.model)
            load_lora(self.model, os.environ['LORA_PATH'])
            self.model.requires_grad_(False)


if __name__ == '__main__':
    evaluator.RecoveryPolicyReader = LoraReader
    evaluator.load_head = load_head
    evaluator.main()
