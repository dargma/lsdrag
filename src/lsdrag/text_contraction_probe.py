"""Bounded continuations from frozen public prefixes; no new policy training."""
import copy
import json
import re
import time

from jsonschema import validate

from .text_decision_benchmark import TOOLS, TextCorpus, canonical
from .text_integrated import ACTIONS, argument_schema, options_for, selection_messages
from .text_integrated_aligned import AlignedReader


def native_action_from_prefix(text):
    text = text.strip()
    if text.startswith('{'):
        return 'FINAL'
    if text.startswith('<tool_call>'):
        match = re.search(r'"name"\s*:\s*"(keyword_search|semantic_search|read_chunk)"', text)
        if match:
            return match.group(1)
    return None


class Budget:
    def __init__(self, seconds=1800, generated=120000, inputs=2000000):
        self.deadline = time.perf_counter()+seconds
        self.generated_cap, self.input_cap = generated, inputs
        self.generated, self.inputs, self.calls = 0, 0, 0

    def reserve(self, inputs=0, generated=512):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError('Global worker wall-time cap')
        if self.generated+generated > self.generated_cap or self.inputs+inputs > self.input_cap:
            raise TimeoutError('Global token cap: reserve before next complete call')

    def charge(self, usage):
        self.generated += usage.get('generated_tokens', usage.get('completion_tokens', 0))
        self.inputs += (usage.get('prompt_tokens', 0) if usage.get('full_prefill_count', 0) else 0) + usage.get('forced_prefix_tokens', 0)
        self.calls += 1
        assert self.generated <= self.generated_cap and self.inputs <= self.input_cap


class ProbeReader(AlignedReader):
    def native_action_cached(self, deadline):
        """Decode to the native name boundary, charging every emitted token.

        Keep the original decision cache unchanged for the common aligned
        argument generator. Routing syntax is therefore charged in addition to
        the aligned forced syntax: a diagnostic selector, not optimized native.
        """
        torch = self.torch
        start = time.perf_counter()
        original = self.pending
        pending = copy.deepcopy(original)
        generated = []
        mask = pending['mask']
        self._sync()
        with torch.inference_mode():
            output = self.model.lm_head(pending['last_hidden']).unsqueeze(1)
            past = pending['past']
            action = None
            for _ in range(self.config.max_output_tokens):
                if time.perf_counter() >= deadline:
                    raise TimeoutError('Native selector deadline')
                tid = int(output[0, -1].argmax())
                generated.append(tid)
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                action = native_action_from_prefix(text)
                if action is not None or tid in self.stop_ids:
                    break
                token = torch.tensor([[tid]], device=self.model.device)
                mask = torch.cat([mask, torch.ones_like(token)], -1)
                result = self.model(input_ids=token, attention_mask=mask, past_key_values=past,
                                    use_cache=True, logits_to_keep=1)
                output, past = result.logits, result.past_key_values
        self._sync()
        assert self.pending is original
        assert original['past'].get_seq_length() == len(original['ids'])
        return action, {'prompt_tokens': 0, 'completion_tokens': len(generated),
                        'generated_tokens': len(generated), 'full_prefill_count': 0,
                        'forced_prefix_tokens': 0, 'seconds': time.perf_counter()-start,
                        'token_ids': generated, 'raw_prefix': text, 'cache_clone_count': 1,
                        'original_cache_length': len(original['ids']),
                        'mode': 'native_name_boundary_then_common_aligned_replay'}, None if action else 'native_name_not_parsed'


def continuation(question, rows, vectors, embedder, reader, initial_messages, initial_history,
                 first, budget, tool_cap=2):
    """Every branch recomputes its prefix; future teacher suffix is never read."""
    start = time.perf_counter()
    corpus = TextCorpus(rows, vectors, embedder)
    messages, history = copy.deepcopy(initial_messages), copy.deepcopy(initial_history)
    assert len(history) == 1
    calls, actions, errors = [], [], []
    answer, valid, forced_final, snapshot = {'answer': '', 'support_ids': []}, False, False, None
    for turn in range(tool_cap+1):
        options = options_for(question, history)
        inputs = int(reader.prepare(selection_messages(messages, options), TOOLS)['input_ids'].shape[1])
        context_forced = inputs+reader.config.max_output_tokens+256 > reader.config.context_limit
        budget.reserve(inputs=inputs, generated=0)
        choice, _, usage = reader.capture(messages, options, 'head', budget.deadline)
        budget.charge(usage)
        calls.append({'kind': 'decision', 'usage': usage})
        action = choice['action']
        if turn == tool_cap or context_forced:
            action, forced_final = 'FINAL', True
        elif turn == 0:
            if first == 'native':
                budget.reserve()
                action, usage, error = reader.native_action_cached(budget.deadline)
                budget.charge(usage)
                calls.append({'kind': 'native_selector', 'usage': usage})
                if error:
                    errors.append(error)
                    reader.pending = None
                    break
            elif first != 'head':
                action = first
        legal = {o['action'] for o in options}
        if action not in legal:
            errors.append('illegal_selected_action: '+str(action))
            reader.pending = None
            break
        actions.append({'turn': turn, 'head_action': choice['action'], 'executed_action': action})
        budget.reserve(inputs=64)
        value, usage, failure = reader.continue_cached(action, history, budget.deadline)
        budget.charge(usage)
        calls.append({'kind': 'final' if action == 'FINAL' else 'arguments', 'usage': usage})
        if failure:
            errors.append(('final_format: ' if action == 'FINAL' else 'argument_format: ')+failure)
            break
        if action == 'FINAL':
            answer, valid = value, True
            break
        validate(value, argument_schema(action, history))
        tick = time.perf_counter()
        result = corpus.call(action, value)
        item = {'tool_name': action, 'arguments': value, 'result': result,
                'argument_reused': False, 'seconds': time.perf_counter()-tick}
        history.append(item)
        tc = {'id': f'probe_{turn}', 'type': 'function',
              'function': {'name': action, 'arguments': canonical(value)}}
        messages.extend([{'role': 'assistant', 'content': None, 'tool_calls': [tc]},
                         {'role': 'tool', 'tool_call_id': tc['id'], 'content': canonical(result)}])
        if snapshot is None:
            snapshot = {'messages': copy.deepcopy(messages), 'history': copy.deepcopy(history),
                        'calls': copy.deepcopy(calls), 'seconds': time.perf_counter()-start}
    ep = {'first_policy': first, 'answer': answer['answer'], 'support_ids': answer['support_ids'],
          'answer_format_valid': valid, 'errors': errors, 'reader_calls': calls,
          'actions': actions, 'trajectory': history, 'suffix_tool_calls': len(history)-1,
          'full_tool_calls': len(history), 'decision_turns': sum(c['kind'] == 'decision' for c in calls),
          'forced_final': forced_final, 'seconds': time.perf_counter()-start}
    return ep, snapshot


def answer_after_first_tool(reader, snapshot, budget):
    """B=1 answer is generated from the real intermediate state, never reused."""
    messages, history = snapshot['messages'], snapshot['history']
    start = time.perf_counter()
    inputs = int(reader.prepare(messages, TOOLS)['input_ids'].shape[1])
    budget.reserve(inputs=inputs, generated=0)
    question = next(m['content'] for m in messages if m['role'] == 'user')
    _, _, usage = reader.capture(messages, options_for(question, history), 'head', budget.deadline)
    budget.charge(usage)
    calls = copy.deepcopy(snapshot['calls'])+[{'kind': 'decision', 'usage': usage}]
    budget.reserve()
    value, usage, error = reader.continue_cached('FINAL', history, budget.deadline)
    budget.charge(usage)
    calls.append({'kind': 'final', 'usage': usage})
    return {'answer': value['answer'] if value else '', 'support_ids': value['support_ids'] if value else [],
            'answer_format_valid': error is None, 'errors': [error] if error else [],
            'reader_calls': calls, 'trajectory': history, 'suffix_tool_calls': 1,
            'full_tool_calls': len(history), 'decision_turns': sum(c['kind'] == 'decision' for c in calls),
            'forced_final': True, 'seconds': snapshot['seconds']+time.perf_counter()-start}
