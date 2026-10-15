from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lsdrag.head_experiment import GeneratedChoiceReader, experiment_prompt, train_behavior_heads
from lsdrag.tool_registry import action_names, tool_schemas


def test_coverage_rejects_missing_actions_and_forced_labels(tmp_path):
    tools = tool_schemas('G1', 'search-linked', 'multimodal_full')
    rows = [{'qid': str(i), 'action': 'FINAL', 'selector': 'generated', 'forced_final': False} for i in range(6)]
    result = train_behavior_heads(rows, np.ones((6, 4)), tmp_path, {}, tools, epochs=2, seeds=(1,))
    assert not result['gate']['passed']
    assert not result['runs']
    with pytest.raises(ValueError, match='natural'):
        train_behavior_heads([{**rows[0], 'forced_final': True}], [[1., 0.]], tmp_path, {}, tools)


def test_all_classes_and_standardization_fold(tmp_path):
    tools = tool_schemas('G1', 'search-linked', 'multimodal_full')
    rows, features = [], []
    for index, action in enumerate(action_names(tools)):
        for q in range(5):
            rows.append({'qid': str(index*5+q), 'action': action, 'selector': 'generated', 'forced_final': False,
                         'episode_status': 'failed' if q == 0 else 'ok'})
            features.append(np.eye(6)[index]*2+3)
    result = train_behavior_heads(rows, features, tmp_path, {'test': True}, tools, epochs=4, seeds=(7,))
    assert result['gate']['passed']
    assert result['examples'] == 30  # Includes failed episodes.
    saved = torch.load(result['runs'][0]['checkpoint'], weights_only=True)
    assert saved['format'] == 'jev-direct-head-v1'
    assert tuple(saved['actions']) == action_names(tools)
    mean, scale = saved['normalization']['mean'], saved['normalization']['scale']
    raw = torch.tensor(np.asarray(features), dtype=torch.float32)
    folded = torch.nn.functional.linear(raw, saved['weight'], saved['bias'])
    normalized_weight = saved['weight']*scale[None, :]
    normalized_bias = saved['bias']+(saved['weight']*mean[None, :]).sum(-1)
    normalized = torch.nn.functional.linear((raw-mean)/scale, normalized_weight, normalized_bias)
    torch.testing.assert_close(folded, normalized)
    assert 'argument JSON' in experiment_prompt('base')


def test_teacher_features_captured_before_decoding_and_no_forced_final_sink():
    reader = GeneratedChoiceReader.__new__(GeneratedChoiceReader)
    reader.torch = torch
    reader.mode = 'generated'
    reader.actions = action_names(tool_schemas('G1', 'search-linked', 'multimodal_full'))
    reader.label_ids = list(range(6))
    reader.stop_ids = {9}
    reader.config = SimpleNamespace(context_limit=100)
    reader._sync = lambda: None
    reader.prepare = lambda messages, tools: {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.ones(1, 2, dtype=torch.long)}
    reader._capture_owner = reader
    reader.feature = None
    reader.current_qid, reader.natural_loop = 'question', 0
    states = []
    reader.state_sink = lambda record, feature: states.append((record, feature))
    class Model:
        def __init__(self):
            self.calls = 0
        def __call__(self, **kwargs):
            logits = torch.zeros(1, 1, 10)
            logits[0, 0, 0 if self.calls == 0 else 9] = 10
            reader.feature = torch.full((1, 4), float(self.calls+1))
            self.calls += 1
            return SimpleNamespace(logits=logits, past_key_values=None)
    reader.model = Model()
    reader.processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: 'A'))
    action, probabilities, usage = reader._select([], [])
    assert action == 'keyword_search'
    assert usage['generated_tokens'] == 2
    assert len(probabilities) == 6
    assert len(states) == 1
    np.testing.assert_array_equal(states[0][1], np.ones(4))
    assert states[0][0]['qid'] == 'question'
    assert states[0][0]['forced_final'] is False


def test_teacher_identity_alignment_and_question_gate(tmp_path):
    from lsdrag.head_experiment import validate_teacher_artifact
    artifact = {'feature_contract': {'hash': 'x'}, 'records': [{'qid': 'train', 'loop': 1}], 'features': torch.zeros(1, 4)}
    validate_teacher_artifact(artifact, 'train', {'hash': 'x'})
    with pytest.raises(ValueError, match='qid'):
        validate_teacher_artifact(artifact, 'development', {'hash': 'x'})
    with pytest.raises(ValueError, match='shape'):
        validate_teacher_artifact({**artifact, 'features': torch.zeros(2, 4)}, 'train', {'hash': 'x'})
    tools = tool_schemas('G1', 'search-linked', 'multimodal_full')
    rows = [{'qid': str(q), 'action': action, 'selector': 'generated', 'forced_final': False}
            for q in range(5) for action in action_names(tools)]
    result = train_behavior_heads(rows, np.ones((30, 4)), tmp_path, {}, tools, epochs=1, seeds=(1,))
    assert all(v == 5 for v in result['gate']['distinct_questions_per_action'].values())
    assert not result['gate']['passed']  # Class coverage alone cannot enable training.


def test_tool_infrastructure_vs_policy_errors_and_fixed_prefix():
    from lsdrag.head_experiment import infrastructure_tool_errors, validate_completed_prefix
    policy = [{'result': {'error': message}} for message in
              ['ValueError: Unknown chunk ID: missing', 'ValueError: Provide 1–8 figure IDs',
               'ValidationError: query is required']]
    assert not infrastructure_tool_errors(policy)
    failures = [{'result': {'error': message}} for message in
                ['RuntimeError: worker exited', 'OSError: disk failure', 'KeyError: bad index',
                 'ValueError: Index does not match corpus', 'EOFError: transport closed']]
    assert len(infrastructure_tool_errors(policy+failures)) == 5
    train = [str(i) for i in range(100)]
    validate_completed_prefix(train[:60], train)
    validate_completed_prefix(train[:80], train)
    with pytest.raises(ValueError, match='fixed first'):
        validate_completed_prefix(train[20:80], train)
    with pytest.raises(ValueError, match='fixed first'):
        validate_completed_prefix(train[:61], train)


def test_selection_suffix_keeps_history_and_binds_all_controller_prefixes(tmp_path):
    import hashlib
    from lsdrag.agent import AgentConfig
    from lsdrag.head_experiment import selection_messages, SELECTION_REQUEST, build_head_feature_contract
    history = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': [{'type': 'text', 'text': 'question'}]}]
    prepared = selection_messages(history)
    assert len(history) == 2
    assert prepared[:-1] == history
    assert prepared[-1] == {'role': 'user', 'content': SELECTION_REQUEST}
    prepared[1]['content'][0]['text'] = 'mutated'
    assert history[1]['content'][0]['text'] == 'question'
    path = tmp_path/('a'*40)
    path.mkdir()
    (path/'config.json').write_text('{}')
    (path/'tokenizer_config.json').write_text('{}')
    tools = tool_schemas('G1', 'search-linked', 'multimodal_full')
    contract = build_head_feature_contract(path, AgentConfig(), tools, [hashlib.sha256(b'system').hexdigest()])
    selector = contract['selector_preprocessing']
    assert selector['tools_preserved']
    assert selector['applies_to'] == ['generated', 'first_token', 'head']
    assert selector['instruction_sha256'] == hashlib.sha256(SELECTION_REQUEST.encode()).hexdigest()
    assert len(contract['actions']) == 6
