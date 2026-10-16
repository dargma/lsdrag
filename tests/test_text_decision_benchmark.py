import json
import numpy as np
import pytest

from lsdrag.text_decision_benchmark import public_paragraphs, execution_options, TextCorpus, run_episode


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


def test_native_final_uses_new_answer_contract():
    class Reader:
        def chat(self, messages, tools):
            assert len(tools)==3 and all(isinstance(m['content'],str) for m in messages)
            return {'message':{'content':json.dumps({'answer':'Leeds','support_ids':[]})},'finish_reason':'stop'}, {'prompt_tokens':10,'completion_tokens':5}
    result=run_episode('Where?',[],np.zeros((0,768)),None,Reader(),'Text only','native')
    assert result['answer']=='Leeds' and result['answer_format_valid']
    assert result['tool_calls']==0 and result['completion_tokens']==5


def test_reuse_dispatch_skips_argument_generation(monkeypatch):
    import lsdrag.text_decision_benchmark as module
    class D3:
        count=0
        def choose(self,messages,options):
            self.count+=1
            if self.count==1:
                o=next(o for o in options if o['action']=='semantic_search' and o['arguments'])
            elif self.count==2:
                o=next(o for o in options if o['action']=='read_chunk' and o['arguments'])
            else:o=options[-1]
            return o,{'seconds':0.01,'result':{'usage':{'input_tokens':5,'output_tokens':0}}}
    class Embedder:
        def encode(self,inputs,role):return np.ones((1,768),dtype=np.float32)/np.sqrt(768)
    def generate(reader,messages,action):
        assert action=='FINAL', 'Reuse must not call the argument generator'
        return {'answer':'Leeds','support_ids':['p7']},{'generated_tokens':4},None
    monkeypatch.setattr(module,'generate',generate)
    result=run_episode('Where?',[{'chunk_id':'p7','title':'T','text':'Alice was born in Leeds.'}],
                       np.ones((1,768)),Embedder(),None,'Text only','d3',D3())
    assert result['reused_argument_calls']==2 and result['argument_tokens']==0
    assert result['answer']=='Leeds' and result['tool_calls']==2
