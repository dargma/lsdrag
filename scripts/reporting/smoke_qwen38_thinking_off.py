import json,time,traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
root=Path('/content/local_fast/lsdrag-model-candidates/qwen38-distill')
manifest=json.loads((root/'download_verified.json').read_text());snapshot=manifest['snapshot']
report={'repo':manifest['repo'],'revision':manifest['revision'],'enable_thinking':False,'text_only':True,'tests':[],'quality_benchmark':False}
def save(): (root/'thinking_off_smoke.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
try:
 torch.set_num_threads(2)
 tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
 model=Qwen3_5ForConditionalGeneration.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
 tests=[('answer','What is 2 + 3? Reply with only the numeral.',None),('json','Return only a JSON object with the key "answer" and value "Paris".',None),('tool','Call keyword_search with query "capital of France". Do not answer the question.',[{'type':'function','function':{'name':'keyword_search','description':'Search text by keywords.','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}}}])]
 for name,prompt,tools in tests:
  messages=[{'role':'system','content':'Follow instructions concisely. Do not produce reasoning.'},{'role':'user','content':prompt}]
  rendered=tok.apply_chat_template(messages,tools=tools,tokenize=False,add_generation_prompt=True,enable_thinking=False)
  inputs=tok(rendered,return_tensors='pt').to('cuda');start=time.perf_counter()
  with torch.inference_mode(): out=model.generate(**inputs,max_new_tokens=128,do_sample=False)
  ids=out[0,inputs.input_ids.shape[1]:];text=tok.decode(ids,skip_special_tokens=False)
  r={'name':name,'prompt_tail':rendered[-100:],'closed_empty_think_prefix':rendered.endswith('<think>\n\n</think>\n\n'),'generated':text,'tokens':len(ids),'seconds_observed_shared_gpu':time.perf_counter()-start,'generated_think_tags':'<think>' in text or '</think>' in text,'hit_token_cap':len(ids)>=128}
  if name=='json':
   try:r['json_valid']=json.loads(tok.decode(ids,skip_special_tokens=True))=={'answer':'Paris'}
   except Exception:r['json_valid']=False
  report['tests'].append(r);save();print(json.dumps(r),flush=True)
 report['complete']=True;report['hidden_size']=model.config.text_config.hidden_size;save()
except Exception as e:
 report['error']=repr(e);save();traceback.print_exc();raise
