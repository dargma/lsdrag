"""Check branch isolation, observable IDs and hard caps without loading models."""
import copy
import json
import unittest
from types import SimpleNamespace

import numpy as np

from lsdrag.text_contraction_probe import Budget, answer_after_first_tool, continuation, native_action_from_prefix


class FakeReader:
    config = SimpleNamespace(max_output_tokens=512, context_limit=16384)

    def prepare(self, messages, tools):
        return {'input_ids': np.zeros((1, 100), dtype=int)}

    def capture(self, messages, options, mode, deadline):
        self.pending = True
        return {'action': 'read_chunk'}, None, {'prompt_tokens': 100, 'full_prefill_count': 1}

    def continue_cached(self, action, history, deadline):
        assert self.pending
        self.pending = None
        value = {'answer': 'visible answer', 'support_ids': ['p0']} if action == 'FINAL' else {'chunk_ids': ['p0']}
        return value, {'generated_tokens': 3, 'forced_prefix_tokens': 0}, None


class Checks(unittest.TestCase):
    def setUp(self):
        self.rows = [{'chunk_id': 'p0', 'title': 'public', 'text': 'visible answer'}]
        self.history = [{'tool_name': 'keyword_search', 'arguments': {'keywords': ['visible'], 'top_k': 5},
                         'result': [{'chunk_id': 'p0', 'title': 'public', 'snippet': 'visible'}], 'seconds': 0.0}]
        self.messages = [{'role': 'user', 'content': 'public question'},
                         {'role': 'assistant', 'content': None, 'tool_calls': [
                             {'id': 'initial', 'type': 'function', 'function': {
                                 'name': 'keyword_search', 'arguments': json.dumps(self.history[0]['arguments'])}}]},
                         {'role': 'tool', 'tool_call_id': 'initial', 'content': json.dumps(self.history[0]['result'])}]

    def test_native_names_are_distinguishable(self):
        self.assertIsNone(native_action_from_prefix('<tool_call>\n{"name": "'))
        for action in ['keyword_search', 'semantic_search', 'read_chunk']:
            self.assertEqual(native_action_from_prefix('<tool_call>\n{"name": "'+action+'"'), action)
        self.assertEqual(native_action_from_prefix(' {"answer":'), 'FINAL')

    def test_branch_and_snapshot_isolation(self):
        original = copy.deepcopy(self.history)
        ep, snapshot = continuation('public question', self.rows, np.ones((1, 1)), None,
                                    FakeReader(), self.messages, self.history, 'head', Budget(), tool_cap=1)
        self.assertEqual(ep['suffix_tool_calls'], 1)
        self.assertTrue(ep['forced_final'])
        self.assertEqual(self.history, original)
        self.assertEqual(len(snapshot['history']), 2)
        ep['trajectory'][1]['result'][0]['text'] = 'changed'
        self.assertEqual(snapshot['history'][1]['result'][0]['text'], 'visible answer')
        one = answer_after_first_tool(FakeReader(), snapshot, Budget())
        self.assertEqual(one['answer'], 'visible answer')

    def test_immediate_final_cannot_read_future(self):
        ep, snapshot = continuation('public question', self.rows, np.ones((1, 1)), None,
                                    FakeReader(), self.messages, self.history, 'FINAL', Budget())
        self.assertEqual(ep['suffix_tool_calls'], 0)
        self.assertIsNone(snapshot)
        self.assertEqual(ep['trajectory'], self.history)

    def test_budget_and_unobserved_read(self):
        budget = Budget(generated=3, inputs=100)
        budget.charge({'generated_tokens': 3, 'prompt_tokens': 100, 'full_prefill_count': 1})
        with self.assertRaises(TimeoutError):
            budget.reserve(generated=1)
        history = copy.deepcopy(self.history)
        history[0]['result'] = []
        messages = copy.deepcopy(self.messages)
        messages[-1]['content'] = '[]'
        ep, _ = continuation('public question', self.rows, np.ones((1, 1)), None,
                             FakeReader(), messages, history, 'read_chunk', Budget())
        self.assertIn('illegal_selected_action', ep['errors'][0])
        self.assertEqual(ep['suffix_tool_calls'], 0)


if __name__ == '__main__':
    unittest.main()
