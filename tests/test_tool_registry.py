from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsdrag.tool_registry import (action_names, bind_tools, render_tool_instruction,
    synchronize_prompt, tool_contract_sha256, tool_schemas)


@pytest.mark.parametrize('arm', ['B0','C1','G1'])
@pytest.mark.parametrize('policy', ['agent-read','search-linked'])
def test_legacy_schema_contract(arm,policy):
    tools=tool_schemas(arm,policy)
    assert action_names(tools)==('keyword_search','semantic_search','read_chunk','FINAL')
    keyword=tools[0]['function']
    expected='Find chunks by literal case-insensitive keywords. Use short specific terms or entity names, 1–3 words each. Returns IDs and abbreviated snippets; read promising chunks before answering.'
    if policy=='search-linked':
        expected+=' Image-linked chunks in returned results are automatically read within the shared image budget; their actual images follow before your next response. Use read_chunk for text-only chunks or further evidence.'
    assert keyword['description']==expected
    assert keyword['parameters']=={'type':'object','properties':{
        'keywords':{'type':'array','items':{'type':'string'},'minItems':1},
        'top_k':{'type':'integer','minimum':1,'maximum':20,'default':5}},
        'required':['keywords'],'additionalProperties':False}
    assert tools[2]['function']['parameters']['properties']['chunk_ids']=={
        'type':'array','items':{'type':'string'},'minItems':1,'maxItems':8}


def test_full_profile_and_no_mutable_schema_aliases():
    expected=('keyword_search','semantic_search','read_chunk','image_search','figure_read','FINAL')
    assert action_names(tool_schemas('G1',tool_profile='multimodal_full'))==expected
    assert action_names(tool_schemas('V2'))==expected
    with pytest.raises(ValueError,match='requires G1 or V2'):
        tool_schemas('B0',tool_profile='multimodal_full')
    tools=tool_schemas('V2');tools[0]['function']['parameters']['properties']['top_k']['maximum']=999
    assert tool_schemas('V2')[0]['function']['parameters']['properties']['top_k']['maximum']==20
    assert tool_schemas('V2')[1]['function']['parameters']['properties']['top_k']['maximum']==20


def test_names_and_hash_reject_unknown_duplicate_and_track_order_schema():
    tools=tool_schemas('G1');same=deepcopy(tools)
    assert tool_contract_sha256(tools)==tool_contract_sha256(same)
    assert tool_contract_sha256(tools)!=tool_contract_sha256(list(reversed(tools)))
    same[0]['function']['description']+=' changed'
    assert tool_contract_sha256(tools)!=tool_contract_sha256(same)
    with pytest.raises(ValueError,match='Duplicate'):
        action_names(tools+tools[:1])
    bad=deepcopy(tools);bad[0]['function']['name']='shell'
    with pytest.raises(ValueError,match='Unsupported'):
        action_names(bad)


def test_direct_dispatch_uses_only_advertised_bindings():
    calls=[]
    retrieval=SimpleNamespace(arm='G1',keyword_search=lambda **kw:calls.append(('keyword',kw)),
        semantic_search=lambda **kw:calls.append(('semantic',kw)),image_search=lambda **kw:calls.append(('image',kw)))
    context=SimpleNamespace(read=lambda **kw:calls.append(('read',kw)),figure_read=lambda **kw:calls.append(('figure',kw)))
    schemas=tool_schemas('G1',tool_profile='multimodal_full');handlers=bind_tools(retrieval,context,schemas)
    assert tuple(handlers)==action_names(schemas)[:-1]
    handlers['image_search'](query='bridge',top_k=2);handlers['figure_read'](figure_ids=['id'])
    assert calls==[('image',{'query':'bridge','top_k':2}),('figure',{'figure_ids':['id']})]
    del retrieval.image_search
    with pytest.raises(ValueError,match='no callable'):
        bind_tools(retrieval,context,schemas)


def test_prompt_contract_is_idempotent_and_preserves_answer_rules():
    tools=tool_schemas('G1','search-linked','multimodal_full')
    prompt=(Path(__file__).resolve().parents[1]/'prompts/search_linked.txt').read_text()
    updated=synchronize_prompt(prompt,tools,'search-linked')
    assert '**image_search**' in updated and '**figure_read**' in updated
    assert updated[updated.index('## When Answering'):].startswith(prompt[prompt.index('## When Answering'):].rstrip())
    assert updated.count('<!-- tool-contract:start -->')==1
    assert synchronize_prompt(updated,tools,'search-linked')==updated
    assert tool_contract_sha256(tools) in updated


def test_prompt_rejects_unavailable_visual_strategy_and_wrong_delivery():
    root=Path(__file__).resolve().parents[1]
    visual=(root/'prompts/visual.txt').read_text()
    with pytest.raises(ValueError,match='unavailable tool'):
        synchronize_prompt(visual,tool_schemas('G1'),'agent-read')
    baseline=(root/'prompts/baseline.txt').read_text()
    with pytest.raises(ValueError,match='contradicts search-linked'):
        synchronize_prompt(baseline,tool_schemas('G1','search-linked'),'search-linked')
    linked=(root/'prompts/search_linked.txt').read_text()
    with pytest.raises(ValueError,match='contradicts agent-read'):
        synchronize_prompt(linked,tool_schemas('G1'),'agent-read')
    instruction=render_tool_instruction(tool_schemas('G1'),'agent-read')
    assert 'image_search' not in instruction and 'figure_read' not in instruction
