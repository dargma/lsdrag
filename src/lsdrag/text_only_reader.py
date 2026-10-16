"""Non-thinking-only causal reader and zero-decode execution-choice baseline."""
import copy
import hashlib
import json
from types import SimpleNamespace
import time

from .text_decision_benchmark import canonical


class TextOnlyReader:
    def __init__(self, path, config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.config = torch, config
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.processor = SimpleNamespace(tokenizer=self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
            device_map='cuda', attn_implementation='sdpa', local_files_only=True).eval().requires_grad_(False)
        stop = self.model.generation_config.eos_token_id
        self.stop_ids = set(stop if isinstance(stop, list) else [stop])
        self.labels = list('ABCDEFGHIJKLMNOP')
        self.label_ids = []
        for label in self.labels:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError('Choice labels must each be a single token')
            self.label_ids.append(ids[0])

    def _sync(self):
        self.torch.cuda.synchronize()

    def prepare(self, messages, tools):
        if any(not isinstance(m.get('content'), (str, type(None))) for m in messages):
            raise ValueError('Text-only reader refuses media')
        rendered = self.tokenizer.apply_chat_template(copy.deepcopy(messages), tools=tools or None,
                     add_generation_prompt=True, tokenize=False)
        if any(t in rendered for t in ['<think>', '<|think|>', '<|channel>thought']):
            raise ValueError('Thinking control/prefix detected in non-thinking-only reader prompt')
        return self.tokenizer(rendered, return_tensors='pt').to(self.model.device)

    def choose(self, messages, options):
        if not 1 <= len(options) <= len(self.labels):
            raise ValueError('Expected at most16 execution options')
        request = ('Choose the best next operation to solve the original multi-hop question. '
            'Read useful evidence before answering. FINAL only when evidence is sufficient. '
            'Select reusable arguments when they fit; otherwise choose GENERATE_ARGS. '
            'Return only one choice letter. Options:\n'+
            '\n'.join(f'{self.labels[i]}: {canonical(o)}' for i,o in enumerate(options)))
        start = time.perf_counter()
        inputs = self.prepare(messages+[{'role':'user','content':request}], None)
        if inputs['input_ids'].shape[1]+1 > self.config.context_limit:
            raise ValueError('Decision context exceeds budget')
        self._sync(); tick = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model(**inputs, use_cache=False, logits_to_keep=1)
            logits = out.logits[0,-1,self.label_ids[:len(options)]].float()
            probs = logits.softmax(-1).cpu().tolist()
            index = int(logits.argmax().item())
        self._sync()
        return options[index], {'seconds':time.perf_counter()-start,
            'result':{'answers':{'next':{'choice':self.labels[index],'probabilities':probs}},
                      'usage':{'input_tokens':int(inputs['input_ids'].shape[1]),'output_tokens':0,
                               'prefill_seconds':time.perf_counter()-tick}},
            'request_sha256':hashlib.sha256(request.encode()).hexdigest()}
