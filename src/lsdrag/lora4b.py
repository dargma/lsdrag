"""LoRA on the 4B reader for joint head+LLM GRPO.

Rollouts sample payload tokens (tool arguments and the final answer) at a fixed
temperature and record their behavior log-probs, so the trainer can replay them
teacher-forced with gradients. Greedy decoding (temperature 0) is unchanged.
"""
import json
import time
from types import SimpleNamespace

import torch
from jsonschema import validate

from .decision_policy_recovery import RecoveryPolicyReader
from .text_decision_benchmark import FINAL_SCHEMA
from .text_integrated import argument_schema, advance_json_parser
from .text_integrated_aligned import syntax_prefix

TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


class LoRALinear(torch.nn.Module):
    """base(x) + scale*B(A(x)); A, B in fp32 so small updates are not rounded away in bf16."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.lora_A = torch.nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**.5)
        self.scale = alpha/rank

    def forward(self, x):
        return self.base(x)+((x.float()@self.lora_A.T)@self.lora_B.T*self.scale).to(x.dtype)


def attach_lora(model, rank=16, alpha=32):
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and name.split('.')[-1] in TARGETS]
    for name, module in targets:
        parent = model.get_submodule(name.rsplit('.', 1)[0])
        setattr(parent, name.rsplit('.', 1)[1], LoRALinear(module, rank, alpha))
    return [p for n, p in model.named_parameters() if 'lora_' in n]


def lora_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if 'lora_' in k}


def load_lora(model, path):
    state = torch.load(path, weights_only=True, map_location='cpu')
    missing = model.load_state_dict(state, strict=False)
    assert not missing.unexpected_keys and not [k for k in missing.missing_keys if 'lora_' in k]


class SamplingReader(RecoveryPolicyReader):
    """temperature>0 samples payload tokens; grammar fallbacks are flagged, not trained on."""
    temperature = 0.

    def continue_cached(self, action, trajectory, deadline=float('inf')):
        if not self.temperature:
            return super().continue_cached(action, trajectory, deadline)
        from lmformatenforcer import JsonSchemaParser, TokenEnforcer
        start = time.perf_counter(); pending = self.pending; self.pending = None
        if pending is None:
            raise RuntimeError('Missing decision cache')
        prefix, argprefix = syntax_prefix(action); ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        schema = FINAL_SCHEMA if action == 'FINAL' else argument_schema(action, trajectory)
        parser = advance_json_parser(JsonSchemaParser(schema, self.json_config), argprefix); initial = parser
        context = list(pending['ids']); sequence = context+ids; mask = pending['mask']
        generated, logps, free = [], [], []; decoded = ''; complete = False; grammar_error = None
        if len(sequence)+self.config.max_output_tokens > self.config.context_limit:
            raise ValueError('Context limit')
        with torch.inference_mode():
            if ids:
                token = torch.tensor([ids], device=self.model.device); mask = torch.cat([mask, torch.ones_like(token)], -1)
                output = self.model(input_ids=token, attention_mask=mask, past_key_values=pending['past'], use_cache=True, logits_to_keep=1)
            else:
                output = SimpleNamespace(logits=self.model.lm_head(pending['last_hidden']).unsqueeze(1), past_key_values=pending['past'])
            for _ in range(self.config.max_output_tokens-len(ids)):
                if time.perf_counter() >= deadline:
                    raise TimeoutError('Worker deadline')
                logp = (output.logits[0, -1].float()/self.temperature).log_softmax(-1)
                tid = int(torch.multinomial(logp.exp(), 1)); ok = True
                proposed = self.tokenizer.decode(generated+[tid], skip_special_tokens=True)
                candidate = advance_json_parser(initial, proposed)
                if tid in self.stop_ids or candidate is None or len(proposed) <= len(decoded):
                    allowed = TokenEnforcer(self.grammar_data, parser).get_allowed_tokens(sequence).allowed_tokens
                    tid = allowed[int(torch.multinomial(logp[allowed].exp(), 1))]; ok = False
                    proposed = self.tokenizer.decode(generated+[tid], skip_special_tokens=True)
                    candidate = advance_json_parser(initial, proposed)
                    if candidate is None:
                        grammar_error = 'ValueError: Grammar token mismatch'; break
                generated.append(tid); logps.append(float(logp[tid])); free.append(ok)
                sequence.append(tid); decoded = proposed; parser = candidate
                try:
                    json.loads(argprefix+decoded); complete = True; break
                except json.JSONDecodeError:
                    pass
                token = torch.tensor([[tid]], device=self.model.device); mask = torch.cat([mask, torch.ones_like(token)], -1)
                output = self.model(input_ids=token, attention_mask=mask, past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
        text = argprefix+decoded
        usage = {'prompt_tokens': len(ids), 'completion_tokens': len(generated), 'generated_tokens': len(generated),
                 'forced_prefix_tokens': len(ids), 'seconds': time.perf_counter()-start, 'full_prefill_count': 0,
                 'raw_json': text, 'token_ids': generated, 'prefix_token_ids': ids, 'context_ids': context,
                 'token_logps': logps, 'token_free': free, 'temperature': self.temperature}
        if grammar_error:
            return None, usage, grammar_error
        try:
            if not complete:
                raise ValueError('Truncated generation')
            value = json.loads(text); validate(value, schema); return value, usage, None
        except Exception as e:
            return None, usage, type(e).__name__+': '+str(e)


def segment_logps(model, segment, temperature):
    """Teacher-forced log-probs of one sampled payload, with gradients through LoRA."""
    prompt = segment['context_ids']+segment['prefix_token_ids']; gen = segment['token_ids']
    ids = torch.tensor([prompt+gen], device=model.device)
    hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[0, len(prompt)-1:len(prompt)+len(gen)-1]
    logp = (model.lm_head(hidden).float()/temperature).log_softmax(-1)
    return logp.gather(1, torch.tensor(gen, device=model.device)[:, None]).squeeze(1)
