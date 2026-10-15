import json
import hashlib
from types import SimpleNamespace

import pytest
import torch

from lsdrag.direct_reader import DirectHFReader, DirectOutputError


def _tools():
    return [{'type': 'function', 'function': {'name': name, 'description': 'operation '+name,
             'parameters': {'type': 'object', 'properties': {key: {'type': 'string'}}, 'required': [key], 'additionalProperties': False}}}
            for name, key in [('keyword_search', 'keywords'), ('semantic_search', 'query'), ('read_chunk', 'chunk_ids'), ('image_search', 'query'), ('figure_read', 'figure_ids')]]


def _reader():
    reader = DirectHFReader.__new__(DirectHFReader)
    reader.actions = tuple(t['function']['name'] for t in _tools())+('FINAL',)
    reader.torch = torch
    reader.expected_feature_contract = {'allowed_system_prompt_sha256s': [hashlib.sha256(b'allowed').hexdigest()]}
    reader.contract_sha256 = 'contract'
    reader._contract = lambda tools: 'contract'
    return reader


@pytest.mark.parametrize('action,payload', [('image_search', {'query': 'red building'}), ('figure_read', {'figure_ids': 'figure-2'}), ('FINAL', {'answers': ['Paris']})])
def test_selected_name_is_programmatic_and_costs_sum(action, payload):
    reader = _reader()
    reader._select = lambda messages, tools: (action, [0.]*6, {'prefill_seconds': 2., 'head_seconds': .01, 'prompt_tokens': 11, 'generated_tokens': 0})
    instructions = []
    def generate(messages, instruction):
        instructions.append(instruction)
        return json.dumps(payload), {'prefill_seconds': 3., 'decode_seconds': 1., 'prompt_tokens': 21, 'generated_tokens': 4, 'completion_tokens': 4}
    reader._generate_json = generate
    choice, usage = reader.chat([{'role': 'system', 'content': 'allowed'}], _tools())
    if action == 'FINAL':
        assert 'tool_calls' not in choice['message']
    else:
        call = choice['message']['tool_calls'][0]
        assert call['function']['name'] == action
        assert json.loads(call['function']['arguments']) == payload
        assert 'Do not include a function name' in instructions[0]
    assert usage['prefill_seconds'] == 5.
    assert usage['prompt_tokens'] == 32
    assert usage['generated_tokens'] == 4
    assert usage['kv_reused'] is False


def test_bad_arguments_fail_closed_and_final_bypasses_selector():
    reader = _reader()
    reader._select = lambda messages, tools: ('image_search', [], {})
    reader._generate_json = lambda messages, instruction: ('{"tool":"figure_read"}', {})
    with pytest.raises(DirectOutputError) as error:
        reader.chat([{'role': 'system', 'content': 'allowed'}], _tools())
    assert error.value.usage['selected_action'] == 'image_search'
    reader._select = lambda *args: (_ for _ in ()).throw(AssertionError('Must bypass'))
    reader._generate_json = lambda *args: ('{"answers":[]}', {})
    assert reader.chat([{'role': 'system', 'content': 'allowed'}], None)[1]['selected_action'] == 'FINAL'


def test_reject_legacy_order_and_wrong_schema_hash():
    reader = _reader()
    checkpoint = {'format': 'jev-direct-head-v1', 'feature_contract': reader.expected_feature_contract, 'actions': reader.actions, 'tool_contract_sha256': 'contract', 'weight': torch.zeros(6, 8), 'bias': torch.zeros(6)}
    reader._validate_checkpoint(checkpoint)
    with pytest.raises(ValueError, match='action order'):
        reader._validate_checkpoint({**checkpoint, 'actions': reader.actions[:3]+('FINAL',)})
    with pytest.raises(ValueError, match='contract hash'):
        reader._validate_checkpoint({**checkpoint, 'tool_contract_sha256': 'old-schema'})


def test_backbone_readout_does_not_call_vocabulary_head():
    reader = _reader()
    reader.torch = torch
    reader.config = SimpleNamespace(context_limit=100)
    reader.prepare = lambda messages, tools: {'input_ids': torch.tensor([[1, 2]])}
    reader._sync = lambda: None
    class Model:
        def model(self, **kwargs):
            assert kwargs['return_dict'] and kwargs['use_cache']
            return SimpleNamespace(last_hidden_state=torch.tensor([[[0., 0.], [1., 2.]]]))
        def __call__(self, *args, **kwargs):
            raise AssertionError('Vocabulary projection must not run')
    reader.model = Model()
    reader.head = torch.nn.Linear(2, 6)
    with torch.no_grad():
        reader.head.weight.zero_()
        reader.head.bias.copy_(torch.tensor([0., 0., 0., 9., 0., 0.]))
    action, probabilities, usage = reader._select([], _tools())
    assert action == 'image_search'
    assert len(probabilities) == 6
    assert usage['generated_tokens'] == 0
    assert usage['vocab_projection'] == 'none'


def test_feature_prompt_nonfinite_and_truncation_guards():
    reader = _reader()
    checkpoint = {'format': 'jev-direct-head-v1', 'feature_contract': reader.expected_feature_contract,
                  'actions': reader.actions, 'tool_contract_sha256': 'contract',
                  'weight': torch.zeros(6, 8), 'bias': torch.zeros(6)}
    with pytest.raises(ValueError, match='feature contract'):
        reader._validate_checkpoint({**checkpoint, 'feature_contract': {'wrong': True}})
    with pytest.raises(ValueError, match='Nonfinite'):
        reader._validate_checkpoint({**checkpoint, 'weight': torch.full((6, 8), float('nan'))})
    with pytest.raises(ValueError, match='System prompt'):
        reader.chat([{'role': 'system', 'content': 'unapproved'}], None)
    reader._select = lambda *args: ('image_search', [], {'prefill_seconds': 2.})
    reader._generate_json = lambda *args: ('{"query":"building"}', {'finish_reason': 'length'})
    with pytest.raises(DirectOutputError, match='Truncated'):
        reader.chat([{'role': 'system', 'content': 'allowed'}], _tools())
    reader._generate_json = lambda *args: (_ for _ in ()).throw(RuntimeError('context/model failure'))
    with pytest.raises(DirectOutputError) as error:
        reader.chat([{'role': 'system', 'content': 'allowed'}], _tools())
    assert error.value.usage['generation_cost_unknown']
    assert error.value.usage['prefill_seconds'] == 2.


def test_pinned_feature_contract_tracks_assets_and_prompt_allowlist(tmp_path):
    from lsdrag.direct_contract import build_feature_contract
    snapshot = tmp_path/('a'*40)
    snapshot.mkdir()
    (snapshot/'config.json').write_text('{}')
    (snapshot/'tokenizer_config.json').write_text('{}')
    config = SimpleNamespace(image_soft_tokens=1120, context_limit=32768)
    prompt_hash = hashlib.sha256(b'allowed').hexdigest()
    original = build_feature_contract(snapshot, config, _tools(), [prompt_hash])
    assert original['model_revision'] == 'a'*40
    (snapshot/'tokenizer_config.json').write_text('{"changed":true}')
    assert original != build_feature_contract(snapshot, config, _tools(), [prompt_hash])
    with pytest.raises(ValueError, match='nonempty'):
        build_feature_contract(snapshot, config, _tools(), [])
    with pytest.raises(ValueError, match='40-hex'):
        build_feature_contract(tmp_path, config, _tools(), [prompt_hash])


def test_synthetic_checkpoint_rejected_without_diagnostic_before_gpu_load(tmp_path):
    path = tmp_path/'synthetic.pt'
    torch.save({'purpose': 'synthetic_dispatch_smoke'}, path)
    with pytest.raises(ValueError, match='Synthetic diagnostic checkpoint'):
        DirectHFReader('/not/a/model', None, path, _tools())
