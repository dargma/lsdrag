import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from lsdrag.agent import Agent, AgentConfig
from lsdrag.context import EvidenceContext
from lsdrag.direct_reader import DirectHFReader
from lsdrag.tool_registry import action_names, synchronize_prompt, tool_contract_sha256, tool_schemas


def test_all_five_tools_dispatch_and_pixels_reach_next_decision(tmp_path):
    image = tmp_path/'figure.png'
    Image.new('RGB', (8, 8), 'red').save(image)
    figure = {'chunk_id': 'visual', 'image_path': str(image),
              'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest()}
    chunks = {key: {'document_id': 'doc', 'page_index': 0, 'kind': 'image_caption' if key == 'visual' else 'text',
                   'text': key, 'figure_ids': ['f'] if key == 'visual' else []} for key in ['visual', 'text']}
    executed = []

    def handler(name, result):
        def call(**kwargs):
            executed.append((name, kwargs))
            return result
        return call

    retrieval = SimpleNamespace(arm='G1', chunks=chunks, figures={'f': figure},
        keyword_search=handler('keyword_search', []),
        semantic_search=handler('semantic_search', [{'chunk_id': 'visual'}]),
        image_search=handler('image_search', [{'chunk_id': 'visual', 'figure_id': 'f'}]))
    config = AgentConfig(tool_profile='multimodal_full', max_loops=8)
    tools = tool_schemas('G1', config.image_delivery_policy, config.tool_profile)
    prompt = Path('prompts/search_linked.txt').read_text()
    effective = synchronize_prompt(prompt, tools, config.image_delivery_policy)
    reader = DirectHFReader.__new__(DirectHFReader)
    reader.actions = action_names(tools)
    reader.contract_sha256 = tool_contract_sha256(tools)
    reader.expected_feature_contract = {'allowed_system_prompt_sha256s': [hashlib.sha256(effective.encode()).hexdigest()]}
    operations = iter(zip(reader.actions, [
        {'keywords': ['object']}, {'query': 'object'}, {'chunk_ids': ['text']},
        {'query': 'red object'}, {'figure_ids': ['f']},
        {'answers': ['red'], 'citations': ['visual'], 'reasoning': 'Observed actual pixels.'}]))
    selected = {}

    def select(messages, offered):
        assert action_names(offered) == reader.actions
        action, payload = next(operations)
        selected.update(action=action, payload=payload)
        if action in {'read_chunk', 'FINAL'}:
            assert any(isinstance(m.get('content'), list) and any(b.get('type') == 'image_url'
                for b in m['content']) for m in messages)
        return action, [0.]*6, {'prompt_tokens': 5, 'generated_tokens': 0}

    reader._select = select
    reader._generate_json = lambda messages, instruction: (json.dumps(selected['payload']),
        {'prompt_tokens': 7, 'generated_tokens': 2, 'completion_tokens': 2, 'finish_reason': 'stop'})
    result = Agent(retrieval, prompt, config, reader).run('synthetic', 'What color is the object?')
    assert result['status'] == 'ok' and result['answers'] == ['red']
    assert [step['tool_name'] for step in result['trajectory']] == list(reader.actions[:-1])
    assert [name for name, _ in executed] == ['keyword_search', 'semantic_search', 'image_search']
    assert result['read_chunk_ids'] == ['text', 'visual'] and result['delivered_figure_ids'] == ['f']
    assert result['tool_contract_sha256'] == reader.contract_sha256
    assert result['prompt_tokens'] == 6*12
    assert all(call['usage']['direct_dispatch'] for call in result['reader_calls'])


def test_figure_read_spends_budget_on_selected_figure_only():
    retrieval = SimpleNamespace(arm='G1', chunks={'c': {'document_id': 'd', 'page_index': 0,
        'kind': 'image_caption', 'text': 'linked text', 'figure_ids': ['unselected', 'selected']}},
        figures={f: {'chunk_id': 'c', 'image_sha256': f, 'image_path': f} for f in ['unselected', 'selected']})
    context = EvidenceContext(retrieval, max_images=1)
    result = context.figure_read(['selected'])
    assert result[0]['text'] == 'linked text'
    assert result[0]['figures'] == [{'figure_id': 'selected', 'status': 'queued_for_reader'}]
    assert context.opened_figures == {'selected'}
