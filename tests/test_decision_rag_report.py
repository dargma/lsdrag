import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('report',Path(__file__).resolve().parents[1]/'scripts/report_decision_rag.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def test_read_support_proxy_does_not_assert_sufficiency_and_costs_stay_separate():
    row={'qid':'q','policy':'first_token','question':'Q','answers':[], 'em':0,'f1':0,
         'status':'ok','stop_reason':None,'answer_format_valid':True,'errors':[],
         'read_chunk_ids':['c'],'delivered_figure_ids':[],'unique_delivered_images':0,
         'trajectory':[], 'prompt_tokens':30,'completion_tokens':5,'seconds':1,
         'reader_calls':[{'usage':{'selected_action':'FINAL',
             'decision_usage':{'prompt_tokens':10,'generated_tokens':0},
             'generation_usage':{'prompt_tokens':20,'generated_tokens':5}}}]}
    result=report.analyze(row,{'answers':['A'],'support_document_ids':['alias']},
                          {'c':{'document_id':'other','document_ids':['alias']}},{})
    assert result['support_documents_read']==1
    assert any('NOT_proof' in s for s in result['observations'])
    assert result['cost_components']['decision_prompt_tokens']==10
    assert result['cost_components']['final_generated_tokens']==5
    assert not result['cost_components'].get('arguments_generated_tokens')
