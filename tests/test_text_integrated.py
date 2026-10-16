import json
import pytest
from jsonschema import validate, ValidationError

from lsdrag.text_integrated import options_for, target_slot, native_schema, argument_schema, train_heads, candidate_refs, selection_messages


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


def test_compact_refs_preserve_slot_order_without_duplicate_arguments():
    question='Long original question about two fictional places'
    args={'keywords':['RareEntityThatMustNotBeRepeated']}
    result=[{'chunk_id':'p7'},{'chunk_id':'p2'}]
    history=[{'tool_name':'keyword_search','arguments':args,'result':result}]
    messages=[{'role':'user','content':question},
        {'role':'assistant','content':None,'tool_calls':[{'function':{'name':'keyword_search','arguments':json.dumps(args)}}]},
        {'role':'tool','content':json.dumps(result)}]
    options=options_for(question,history)
    refs=candidate_refs(messages,options)
    assert refs['keyword_search']==['NEW','C0']
    assert refs['semantic_search']==['NEW','Q']
    assert refs['read_chunk']==['NEW','R0[:8]','R0[0]','R0[1]']
    text=selection_messages(messages,options)[-1]['content']
    assert question not in text and 'RareEntityThatMustNotBeRepeated' not in text
    for action in refs:
        assert len(refs[action])==sum(o['action']==action for o in options)


def test_speed_gate_rejects_slow_fallback_and_format_failures():
    import copy
    from pathlib import Path
    import runpy
    summary=runpy.run_path(str(Path(__file__).parents[1]/'scripts/run_text_integrated_pilot.py'))['cost_summary']
    rows=[{'qid':str(i),'bucket':bucket,'conditions':{p:{'seconds':2 if p=='native' else 1,'failure':None}
          for p in ['native','integrated','always_generate']}} for i in range(4) for bucket in ['reuse','generate','FINAL']]
    assert summary(rows)['passed']
    slow=copy.deepcopy(rows)
    for row in slow:
        if row['bucket']=='generate':row['conditions']['integrated']['seconds']=3
    assert not summary(slow)['passed']
    rows[0]['conditions']['native']['failure']='invalid native output'
    assert not summary(rows)['passed']
