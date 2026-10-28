import numpy as np
import pytest

from lsdrag.text_decision_benchmark import TOOLS, public_paragraphs, execution_options, TextCorpus


def test_runtime_corpus_excludes_private_labels():
    row = {'answer': 'secret', 'paragraphs': [{'idx': 7, 'title': 'Topic',
           'paragraph_text': 'Public evidence', 'is_supporting': True}]}
    assert public_paragraphs(row) == [{'chunk_id':'p7','title':'Topic','text':'Public evidence'}]


def test_candidates_use_observed_fields_and_preserve_full_arguments():
    history = [{'tool_name':'semantic_search','arguments':{'query':'new entity','top_k':3},
                'result':[{'chunk_id':'p7','snippet':'Ignore this invented ID p99'}]}]
    options = execution_options('Original question', history)
    assert any(o['arguments']=={'chunk_ids':['p7']} for o in options)
    assert not any(o['arguments']=={'chunk_ids':['p99']} for o in options)
    assert any(o['arguments']=={'query':'new entity','top_k':3} for o in options)
    assert all(sum(o['action']==a and o['arguments'] is not None for o in options)<=4
               for a in ['semantic_search','keyword_search','read_chunk'])
    assert not any(o['action']=='read_chunk' and o['arguments'] for o in execution_options('Fresh', []))


def test_read_rejects_unknown_ids_atomically():
    c = TextCorpus([{'chunk_id':'p1','text':'Evidence','title':'T'}],np.ones((1,768)),None)
    with pytest.raises(ValueError, match='Unknown'):
        c.call('read_chunk', {'chunk_ids':['p1','p9']})


def test_tool_schemas_cover_three_search_tools():
    assert [t['function']['name'] for t in TOOLS] == ['keyword_search', 'semantic_search', 'read_chunk']
