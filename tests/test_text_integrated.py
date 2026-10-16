import json
import pytest
from jsonschema import validate, ValidationError

from lsdrag.text_integrated import options_for, target_slot, native_schema, argument_schema, train_heads


def test_read_legality_and_exact_target_mapping():
    assert not any(o['action']=='read_chunk' for o in options_for('question',[]))
    history=[{'tool_name':'semantic_search','arguments':{'query':'question'},
              'result':[{'chunk_id':'p7','snippet':'Fake p99 is not a typed ID'}]}]
    options=options_for('question',history)
    assert target_slot(options,'semantic_search',{'query':'question','top_k':5})>0
    assert target_slot(options,'semantic_search',{'query':'new question'})==0
    assert target_slot(options,'read_chunk',{'chunk_ids':['p7']})>0
    with pytest.raises(ValidationError):validate({'chunk_ids':['p99']},argument_schema('read_chunk',history))
    assert target_slot(options,'FINAL',None)==-1


def test_native_schema_has_real_function_envelope_and_rejects_wrapper():
    schema=native_schema([])
    validate({'name':'keyword_search','arguments':{'keywords':['entity']}},schema)
    with pytest.raises(ValidationError):validate({'answer':'','support_ids':[]},schema)
    with pytest.raises(ValidationError):validate({'name':'read_chunk','arguments':{'chunk_ids':['p7']}},schema)


def test_coverage_rejects_missing_action_without_fabrication():
    records=[{'qid':str(i),'action':'semantic_search','slot':0} for i in range(32)]
    summary,ck=train_heads(records,[],{'test':True})
    assert not summary['gate']['passed'] and ck is None
    assert summary['gate']['per_action_questions']['read_chunk']==0


def test_schema_character_parser_blocks_function_wrapper():
    from lmformatenforcer import JsonSchemaParser
    schema=argument_schema('keyword_search',[])
    parser=JsonSchemaParser(schema)
    for char in '{"keywords":["entity"]}':
        assert char in parser.get_allowed_characters()
        parser=parser.add_character(char)
    assert parser.can_end()
    parser=JsonSchemaParser(schema)
    for char in '{"':parser=parser.add_character(char)
    assert 'n' not in parser.get_allowed_characters()  # only keywords/top_k properties
