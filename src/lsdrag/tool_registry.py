"""Single contract for A-RAG tool schemas, action order, prompts and dispatch."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    properties: dict[str, Any]
    required: tuple[str, ...]
    owner: str
    attribute: str
    visual: bool = False


_IDS = {'type':'array','items':{'type':'string'},'minItems':1,'maxItems':8}
_TOP = {'type':'integer','minimum':1,'maximum':20,'default':5}
_QUERY = {'type':'string','description':'A natural language description of the evidence to retrieve.'}
_AUTO = ' Image-linked chunks in returned results are automatically read within the shared image budget; their actual images follow before your next response. Use read_chunk for text-only chunks or further evidence.'
TOOL_SPECS = (
    ToolSpec('keyword_search', 'Find chunks by literal case-insensitive keywords. Use short specific terms or entity names, 1–3 words each. Returns IDs and abbreviated snippets; read promising chunks before answering.',
        {'keywords':{'type':'array','items':{'type':'string'},'minItems':1},'top_k':_TOP}, ('keywords',), 'retrieval','keyword_search'),
    ToolSpec('semantic_search', 'Find chunks by semantic similarity when wording is unknown or keyword search is insufficient. Returns IDs and short snippets; use read_chunk for full evidence.',
        {'query':_QUERY,'top_k':_TOP}, ('query',), 'retrieval','semantic_search'),
    ToolSpec('read_chunk', 'Read full text for chunk IDs from search. Image-caption chunks also load their linked image. Already-read content is not repeated. Follow adjacent IDs when content is incomplete.',
        {'chunk_ids':_IDS}, ('chunk_ids',), 'context','read'),
    ToolSpec('image_search', 'Find figures by actual visual content using a text query. Describe the subject or scene needed, without assuming the answer. Optionally restrict to discovered document IDs. Returns figure IDs and text only; use figure_read to inspect images.',
        {'query':_QUERY,'document_ids':{'type':'array','items':{'type':'string'}},'top_k':_TOP}, ('query',), 'retrieval','image_search', True),
    ToolSpec('figure_read', 'Inspect selected figures and linked text. The actual images follow the tool results. Use this to verify visual properties before answering.',
        {'figure_ids':_IDS}, ('figure_ids',), 'context','figure_read', True),
)


def _policy(policy):
    if policy not in {'agent-read','search-linked'}:
        raise ValueError('Unknown image delivery policy: '+str(policy))


def tool_schemas(arm, image_delivery_policy='agent-read', tool_profile='legacy'):
    """Legacy preserves historical schemas; full enables G1/V2 explicit images."""
    _policy(image_delivery_policy)
    if tool_profile not in {'legacy','multimodal_full'}:
        raise ValueError('Unknown tool profile: '+str(tool_profile))
    if tool_profile=='multimodal_full' and arm not in {'G1','V2'}:
        raise ValueError('multimodal_full requires G1 or V2 visual indexes')
    selected = [s for s in TOOL_SPECS if not s.visual or arm=='V2' or tool_profile=='multimodal_full']
    result=[]
    for spec in selected:
        description=spec.description
        if spec.name in {'keyword_search','semantic_search'} and image_delivery_policy=='search-linked':
            description += _AUTO
        result.append({'type':'function','function':{'name':spec.name,'description':description,
            'parameters':{'type':'object','properties':deepcopy(spec.properties),
                          'required':list(spec.required),'additionalProperties':False}}})
    return result


def action_names(tools):
    known={s.name for s in TOOL_SPECS}; names=[]
    for tool in tools or ():
        try: name=tool['function']['name']
        except (TypeError,KeyError) as error: raise ValueError('Invalid function tool schema') from error
        if tool.get('type')!='function' or name not in known:
            raise ValueError('Unsupported tool: '+str(name))
        if name in names: raise ValueError('Duplicate tool: '+name)
        names.append(name)
    return tuple(names)+('FINAL',)


def tool_contract_sha256(tools):
    payload={'schemas':tools or [],'actions':action_names(tools)}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def bind_tools(retrieval, context, schemas):
    """Return bound callables for exactly the advertised schemas, fail closed."""
    names=action_names(schemas)[:-1]; specs={s.name:s for s in TOOL_SPECS}
    owners={'retrieval':retrieval,'context':context}; handlers={}
    for name in names:
        spec=specs[name]
        if spec.visual and getattr(retrieval,'arm',None) not in {'G1','V2'}:
            raise ValueError('Visual tool requires G1 or V2: '+name)
        method=getattr(owners[spec.owner],spec.attribute,None)
        if not callable(method): raise ValueError('Runtime has no callable binding for '+name)
        handlers[name]=method
    return handlers


def _delivery(policy):
    _policy(policy)
    if policy=='search-linked':
        return ('Image-linked keyword_search and semantic_search results are automatically read within the shared image budget; actual images arrive before the next response. '
                'Use read_chunk for text-only chunks or additional context. An exhausted image budget means the image was not supplied.')
    return ('Search results provide IDs and abbreviated text; use read_chunk for full evidence and linked images. '
            'Images arrive after reading, within the shared image budget.')


def render_tool_instruction(tools, image_delivery_policy='agent-read'):
    names=action_names(tools)[:-1]; _policy(image_delivery_policy)
    lines=['## Available Tools','']
    lines += [f"- **{tool['function']['name']}**: {tool['function'].get('description','')}" for tool in tools or ()]
    lines += ['', 'Search for the required entity or relationship, inspect supporting evidence, then answer. '
              'Do not infer visual properties from retrieval rank or captions alone.']
    if 'image_search' in names and 'figure_read' in names:
        lines += ['For visual attributes, use image_search with a subject description and figure_read to verify actual pixels. '
                  'Resolve missing entity relationships with text tools before refining visual search.']
    lines += [_delivery(image_delivery_policy)]
    return '\n'.join(lines)


_BLOCK = re.compile(r'\n?<!-- tool-contract:start -->.*?<!-- tool-contract:end -->\n?',re.S)
_SECTION = re.compile(r'^## Available Tools\s*\n.*?(?=^## |\Z)',re.M|re.S)


def synchronize_prompt(prompt, tools, image_delivery_policy='agent-read'):
    """Replace only the tools section; reject contradictory remaining guidance."""
    names=action_names(tools)[:-1]; _policy(image_delivery_policy)
    prompt=_BLOCK.sub('\n',prompt)
    sections=list(_SECTION.finditer(prompt))
    if len(sections)>1: raise ValueError('Prompt contains multiple Available Tools sections')
    remainder=_SECTION.sub('',prompt)
    for spec in TOOL_SPECS:
        if spec.name not in names and re.search(r'\b'+re.escape(spec.name)+r'\b',remainder):
            raise ValueError('Prompt refers to unavailable tool outside Available Tools: '+spec.name)
    if image_delivery_policy=='agent-read' and re.search(r'automatically read|\bauto_read\b',remainder,re.I):
        raise ValueError('Prompt automatic image delivery contradicts agent-read policy')
    if image_delivery_policy=='search-linked' and re.search(
        r'abbreviated text only|images arrive.*when requested by|use read_chunk before answering',remainder,re.I):
        raise ValueError('Prompt explicit-only image delivery contradicts search-linked policy')
    instruction=render_tool_instruction(tools,image_delivery_policy)
    if sections: prompt=_SECTION.sub(lambda _:instruction+'\n\n',prompt,count=1)
    else: prompt=prompt.rstrip()+'\n\n'+instruction+'\n'
    contract={'tool_names':list(names),'actions':list(action_names(tools)),
              'image_delivery_policy':image_delivery_policy,'sha256':tool_contract_sha256(tools)}
    return prompt.rstrip()+'\n\n<!-- tool-contract:start -->\n'+json.dumps(contract,sort_keys=True)+'\n<!-- tool-contract:end -->\n'
