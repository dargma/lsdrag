"""Registry-bound action readout and programmatic tool dispatch metadata.

The backbone is frozen. Action selection performs a full multimodal prefill;
argument/answer generation uses a separate, explicitly charged prefill. This
implementation makes no latency or novelty claim.
"""
import copy
import json
import time

from jsonschema import validate

from .hf_reader import HFReader
from .direct_contract import build_feature_contract, verify_system_prompts


class DirectOutputError(ValueError):
    def __init__(self, message, usage):
        super().__init__(message)
        self.usage = usage


class DirectHFReader(HFReader):
    def __init__(self, model_path, config, head_path, tools, *, diagnostic=False):
        import torch
        self.torch = torch
        self.actions = tuple(tool['function']['name'] for tool in tools)+('FINAL',)
        if len(set(self.actions)) != len(self.actions):
            raise ValueError('Duplicate/reserved registry action')
        self.registry_tools = copy.deepcopy(tools)
        self.contract_sha256 = self._contract(tools)
        self.call_counter = 0
        checkpoint = torch.load(head_path, map_location='cpu', weights_only=True)
        if checkpoint.get('purpose') == 'synthetic_dispatch_smoke' and not diagnostic:
            raise ValueError('Synthetic diagnostic checkpoint cannot be used in production')
        prompts = checkpoint.get('feature_contract', {}).get('allowed_system_prompt_sha256s', [])
        self.expected_feature_contract = build_feature_contract(model_path, config, tools, prompts)
        self._validate_checkpoint(checkpoint)
        # Reject incompatible checkpoints before loading any GPU backbone weights.
        super().__init__(model_path, config, head_path=None)
        hidden_width = self.model.config.get_text_config().hidden_size
        if checkpoint['weight'].shape[1] != hidden_width:
            raise ValueError('Checkpoint hidden width differs from backbone')
        self.head = self.torch.nn.Linear(checkpoint['weight'].shape[1], len(self.actions)).to(self.model.device).float()
        self.head.load_state_dict({key: checkpoint[key] for key in ('weight', 'bias')})
        self.head.eval().requires_grad_(False)

    @staticmethod
    def _contract(tools):
        from .tool_registry import tool_contract_sha256
        return tool_contract_sha256(tools)

    def _validate_checkpoint(self, checkpoint):
        if checkpoint.get('format') != 'jev-direct-head-v1':
            raise ValueError('Unsupported direct-head checkpoint format')
        if checkpoint.get('feature_contract') != self.expected_feature_contract:
            raise ValueError('Checkpoint feature contract mismatch')
        if tuple(checkpoint.get('actions', ())) != self.actions:
            raise ValueError('Checkpoint action order differs from complete tool registry')
        if checkpoint.get('tool_contract_sha256') != self.contract_sha256:
            raise ValueError('Checkpoint tool contract hash mismatch')
        if checkpoint['weight'].ndim != 2 or checkpoint['weight'].shape[0] != len(self.actions):
            raise ValueError('Checkpoint head dimensions do not match registry')
        if checkpoint['bias'].shape != (len(self.actions),):
            raise ValueError('Checkpoint bias dimensions do not match registry')
        if not self.torch.isfinite(checkpoint['weight']).all() or not self.torch.isfinite(checkpoint['bias']).all():
            raise ValueError('Nonfinite direct-head weights')

    def _select(self, messages, tools):
        torch = self.torch
        preprocess_start = time.perf_counter()
        inputs = self.prepare(messages, tools)
        self._sync()
        preprocess_seconds = time.perf_counter()-preprocess_start
        if inputs['input_ids'].shape[1]+1 > self.config.context_limit:
            raise ValueError('Decision context exceeds fixed budget')
        tick = time.perf_counter()
        with torch.inference_mode():
            output = self.model.model(**inputs, use_cache=True, return_dict=True)
            hidden = output.last_hidden_state[:, -1, :]
        self._sync()
        prefill_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        with torch.inference_mode():
            logits = self.head(hidden.float())[0]
            legal_names = {tool['function']['name'] for tool in tools}|{'FINAL'}
            mask = torch.tensor([action in legal_names for action in self.actions], device=logits.device)
            logits = logits.masked_fill(~mask, -torch.inf)
            probs = logits.softmax(-1).cpu().tolist()
            action = self.actions[int(logits.argmax().item())]
        self._sync()
        usage = {'preprocess_seconds': preprocess_seconds, 'prefill_seconds': prefill_seconds,
                 'head_seconds': time.perf_counter()-tick, 'prompt_tokens': int(inputs['input_ids'].shape[1]),
                 'generated_tokens': 0, 'completion_tokens': 0, 'decode_seconds': 0.,
                 'forced_prefix_tokens': 0, 'vocab_projection': 'none'}
        return action, probs, usage

    def _generate_json(self, messages, instruction):
        messages = copy.deepcopy(messages)+[{'role': 'user', 'content': instruction}]
        start = time.perf_counter()
        inputs = self.prepare(messages, None)
        self._sync()
        preprocess = time.perf_counter()-start
        if inputs['input_ids'].shape[1]+self.config.max_output_tokens > self.config.context_limit:
            raise ValueError('Argument/answer context exceeds fixed budget')
        tick = time.perf_counter()
        generated = []
        torch = self.torch
        with torch.inference_mode():
            output = self.model(**inputs, use_cache=True, logits_to_keep=1)
            self._sync()
            prefill = time.perf_counter()-tick
            tick = time.perf_counter()
            mask = inputs['attention_mask']
            for _ in range(self.config.max_output_tokens):
                token = output.logits[:, -1, :].argmax(-1, keepdim=True)
                value = int(token.item())
                generated.append(value)
                if value in self.stop_ids:
                    break
                mask = torch.cat([mask, torch.ones_like(token)], -1)
                output = self.model(input_ids=token, attention_mask=mask, past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
        self._sync()
        decode = time.perf_counter()-tick
        text = self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text, {'preprocess_seconds': preprocess, 'prefill_seconds': prefill,
                      'decode_seconds': decode, 'prompt_tokens': int(inputs['input_ids'].shape[1]),
                      'generated_tokens': len(generated), 'completion_tokens': len(generated),
                      'forced_prefix_tokens': 0, 'reader_wall_seconds': time.perf_counter()-start,
                      'finish_reason': 'stop' if generated and generated[-1] in self.stop_ids else 'length'}

    def chat(self, messages, tools):
        start = time.perf_counter()
        verify_system_prompts(messages, self.expected_feature_contract)
        if tools and self._contract(tools) != self.contract_sha256:
            raise ValueError('Runtime tools differ from checkpoint-bound registry')
        if tools:
            action, probabilities, decision_usage = self._select(messages, tools)
        else:
            action, probabilities = 'FINAL', None
            decision_usage = {}
        if action == 'FINAL':
            instruction = 'Return only a JSON object with an "answers" list of concise answer strings and optional "citations" list of string IDs and "reasoning" string as requested by the system instruction. No tool call or markdown. Use an empty answers list if evidence is insufficient.'
        else:
            selected_tool = next(tool['function'] for tool in tools if tool['function']['name'] == action)
            schema = selected_tool['parameters']
            instruction = ('The runtime has already selected '+action+': '+selected_tool.get('description', '')+'. Return ONLY its argument JSON object conforming to this schema. Do not include a function name, tool-call wrapper, explanation, or markdown. JSON Schema: '+json.dumps(schema, ensure_ascii=False))
        try:
            text, generation_usage = self._generate_json(messages, instruction)
        except Exception as error:
            failure_usage = dict(decision_usage)
            failure_usage.update(selected_action=action, decision_usage=decision_usage,
                                 reader_wall_seconds=time.perf_counter()-start,
                                 generation_cost_unknown=True,
                                 unknown_cost_note='Generation failed before returning detailed accounting; elapsed wall includes attempted generation.')
            raise DirectOutputError(f'{action} generation failed; refusing dispatch: {error}', failure_usage) from error
        usage = {key: decision_usage.get(key, 0)+generation_usage.get(key, 0) for key in
                 ('preprocess_seconds', 'prefill_seconds', 'decode_seconds', 'prompt_tokens', 'generated_tokens', 'completion_tokens', 'forced_prefix_tokens')}
        usage.update(head_seconds=decision_usage.get('head_seconds', 0.), selected_action=action,
                     action_probabilities=probabilities, tool_contract_sha256=self.contract_sha256,
                     decision_usage=decision_usage, generation_usage=generation_usage,
                     reader_wall_seconds=time.perf_counter()-start, kv_reused=False,
                     direct_dispatch=True, raw_json=text)
        try:
            if action != 'FINAL' and generation_usage.get('finish_reason', 'stop') != 'stop':
                raise ValueError('Truncated argument generation cannot dispatch a tool')
            value = json.loads(text)
            if action == 'FINAL':
                validate(value, {'type': 'object', 'properties': {'answers': {'type': 'array', 'items': {'type': 'string'}}, 'citations': {'type': 'array', 'items': {'type': 'string'}}, 'reasoning': {'type': 'string'}}, 'required': ['answers'], 'additionalProperties': False})
                message = {'role': 'assistant', 'content': json.dumps(value, ensure_ascii=False)}
            else:
                validate(value, schema)
                self.call_counter = getattr(self, 'call_counter', 0)+1
                message = {'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': f'direct_call_{self.call_counter}', 'type': 'function', 'function': {
                        'name': action, 'arguments': json.dumps(value, ensure_ascii=False)}}]}
        except Exception as error:
            raise DirectOutputError(f'Invalid {action} JSON; refusing tool dispatch: {error}', usage) from error
        return {'message': message, 'finish_reason': generation_usage.get('finish_reason', 'stop')}, usage
