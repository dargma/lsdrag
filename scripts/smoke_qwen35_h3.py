import json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from lsdrag.decision_qwen35_reader import Qwen35Reader
from lsdrag.text_decision_benchmark import canonical
REG=Path('/content/local_fast/lsdrag-code/lab/registry/qwen35_9b_h3_20261024')
p=json.loads((REG/'download_status.json').read_text())['path'];torch.set_num_threads(2)
r=Qwen35Reader(p,SimpleNamespace(max_output_tokens=512,context_limit=16384));print('loaded',flush=True)
messages=[{'role':'system','content':Path('/content/local_fast/lsdrag-code/prompts/text_rag_integrated_v5.txt').read_text()},{'role':'user','content':'Evidence: Paris is the capital of France. What is the capital of France?'}]
start=time.perf_counter();pack,usage=r.capture_pack(messages,[],start+180);assert pack.shape==(37,4096)
answer,tail,error=r.continue_cached('FINAL',[],start+180);assert error is None,(error,tail);assert answer['answer'].strip().lower()=='paris',answer
# All three forced argument schemas, including read over observed IDs.
checks=[]
for action in ['keyword_search','semantic_search','read_chunk']:
 history=[{'tool_name':'semantic_search','arguments':{'query':'capital of France','top_k':5},'result':[{'chunk_id':'c123','title':'France','snippet':'Paris is the capital of France.'}]}]
 msgs=messages+[{'role':'assistant','content':None,'tool_calls':[{'id':'call_0','type':'function','function':{'name':'semantic_search','arguments':canonical(history[0]['arguments'])}}]},{'role':'tool','tool_call_id':'call_0','content':canonical(history[0]['result'])}]
 r.capture_pack(msgs,history,time.perf_counter()+180);args,u,err=r.continue_cached(action,history,time.perf_counter()+180);assert err is None,(action,err,u)
 checks.append(dict(action=action,arguments=args,generated=u['generated_tokens']))
report=dict(passed=True,model=p,hidden_shape=list(pack.shape),answer=answer,generated=tail['generated_tokens'],tool_checks=checks,loading_info=r.loading_info,seconds=time.perf_counter()-start,thinking=False,profile='Common A-RAG JSON protocol; official empty nonthinking suffix; not native XML tool baseline')
(REG/'smoke.json').write_text(json.dumps(report,indent=2,default=str)+'\n');print(json.dumps(report,indent=2,default=str))
