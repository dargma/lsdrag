import json

from lsdrag.hf_reader import parse_native, processor_messages


def test_typed_native_arguments_and_multiple_calls():
    raw='<|tool_call>call:keyword_search{keywords:[<|"|>a,b<|"|>,<|"|>c<|"|>],top_k:3}<tool_call|><|tool_call>call:read_chunk{chunk_ids:[<|"|>d:c1<|"|>]}<tool_call|><|tool_response>'
    message=parse_native(raw)
    assert message['content'] is None
    assert len(message['tool_calls'])==2
    assert json.loads(message['tool_calls'][0]['function']['arguments'])=={'keywords':['a,b','c'],'top_k':3}
    assert json.loads(message['tool_calls'][1]['function']['arguments'])=={'chunk_ids':['d:c1']}


def test_processor_adaptation_preserves_history_without_mutation():
    original=[{'role':'assistant','content':None,'tool_calls':[{'function':{'name':'read_chunk','arguments':'{"chunk_ids":["c1"]}'}}]},{'role':'tool','tool_call_id':'x','content':'[{"text":"evidence"}]'}]
    converted=processor_messages(original)
    assert converted[0]['tool_calls'][0]['function']['arguments']=={'chunk_ids':['c1']}
    assert converted[1]['content']==[{'type':'text','text':'[{"text":"evidence"}]'}]
    assert isinstance(original[0]['tool_calls'][0]['function']['arguments'],str)
