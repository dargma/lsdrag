import pytest
from jsonschema import validate, ValidationError
from lmformatenforcer import JsonSchemaParser

from lsdrag.fullft import SampledReader
from lsdrag.text_decision_benchmark import TOOLS
from lsdrag.text_integrated import argument_schema, advance_json_parser, native_schema

PREFIX = {'semantic_search': '{"query": "x", "top_k":', 'keyword_search': '{"keywords": ["x"], "top_k":'}
ARGS = {'semantic_search': {'query': 'x'}, 'keyword_search': {'keywords': ['x']}}


def schemas(action):
    # head arm (continue_cached -> _payload) and native arm (fast_native -> _payload) both use payload_schema
    legacy = SampledReader.__new__(SampledReader)
    v22 = SampledReader.__new__(SampledReader); v22.v22 = True
    return [argument_schema(action, []), legacy.payload_schema(action, []), v22.payload_schema(action, [])]


@pytest.mark.parametrize('action', list(PREFIX))
def test_search_schema_bounds_top_k(action):
    for schema in schemas(action):
        for good in (5, 3, 1, 20):
            validate({**ARGS[action], 'top_k': good}, schema)
        validate(ARGS[action], schema)
        for bad in (50000000, 21, 0, -1, 5.5):
            with pytest.raises(ValidationError):
                validate({**ARGS[action], 'top_k': bad}, schema)


@pytest.mark.parametrize('action', list(PREFIX))
def test_grammar_cannot_run_away_on_top_k(action):
    for schema in schemas(action):
        start = lambda text: advance_json_parser(JsonSchemaParser(schema), PREFIX[action]+text)
        for good in (' 5}', ' 3}', '5}', ' 20}', ' 5 }'):
            assert start(good).can_end()
        for bad in (' 50', ' 21', ' 0', ' 100', ' 5e', ' -1'):
            assert start(bad) is None
        assert not set('0123456789eE.') & set(start(' 5').get_allowed_characters())
        assert not set('0123456789eE.') & set(start(' 20').get_allowed_characters())
        assert set('0123456789') <= set(start(' 1').get_allowed_characters())   # 10..19


def test_prompt_tools_and_other_schemas_unchanged():
    # the tool list rendered into the prompt keeps its original form; only the enforcement schema gains the enum
    for tool in TOOLS:
        assert 'enum' not in tool['function']['parameters']['properties'].get('top_k', {})
    history = [{'result': [{'chunk_id': 'p1'}, {'chunk_id': 'p2'}]}]
    assert argument_schema('read_chunk', history)['properties']['chunk_ids'] == {
        'type': 'array', 'items': {'type': 'string', 'enum': ['p1', 'p2']}, 'minItems': 1, 'maxItems': 8}
    branch = next(b for b in native_schema([])['oneOf'] if b['properties']['name']['enum'] == ['semantic_search'])
    assert branch['properties']['arguments']['properties']['top_k']['enum'] == list(range(1, 21))
