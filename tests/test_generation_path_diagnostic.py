import importlib.util
from pathlib import Path
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('diagnostic',Path(__file__).parents[1]/'scripts/diagnose_text_generation_paths.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Tokenizer:
    def __init__(self,pieces):self.pieces=pieces
    def decode(self,ids,**kwargs):return ''.join(self.pieces[i] for i in ids)

def test_exact_native_boundary_does_not_reencode():
    reader=SimpleNamespace(tokenizer=Tokenizer(['<tool_call>\n{"name":"keyword_search",','"arguments":','{','"keywords":["test"]}}']))
    prefix,arg,oracle=m.prefix_boundary(reader,[0,1,2,3],'keyword_search')
    assert prefix==[0,1,2] and arg=='{' and not oracle

def test_boundary_crossing_is_marked_oracle():
    reader=SimpleNamespace(tokenizer=Tokenizer(['<tool_call>\n{"name":"keyword_search",','"arguments":{"keywords":["secret','"]}}']))
    prefix,arg,oracle=m.prefix_boundary(reader,[0,1,2],'keyword_search')
    assert prefix==[0,1] and oracle and 'secret' in arg

def test_final_uses_no_forced_semantic_prefix():
    reader=SimpleNamespace(tokenizer=Tokenizer(['{"answer":"x","support_ids":[]}']))
    assert m.prefix_boundary(reader,[0],'FINAL')==([], '',False)
