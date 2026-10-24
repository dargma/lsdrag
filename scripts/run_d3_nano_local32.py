"""Fixed external typed decision inference on causal training-state requests."""
import copy,json,os,time,hashlib
from pathlib import Path
import torch,transformers
from transformers import AutoModel
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path('/content/local_fast/lsdrag-code');NAME='decision_d3_nano_local32_20261024'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main():
 m=read(REPO/'lab/registry'/NAME/'manifest.json');out=ROOT/'results'/NAME;assert not (out/'status.json').exists()
 assert sha(Path(m['cases']))==m['cases_sha256']
 for f,h in m['source_sha256'].items():assert sha(REPO/f)==h
 start=time.monotonic();pred=[]
 def phase(name,**kw):save(out/'status.json',dict(phase=name,pid=None if name in ['complete','failed'] else os.getpid(),seconds=time.monotonic()-start,requests=len(pred),**kw))
 phase('loading')
 try:
  torch.set_num_threads(2);torch.manual_seed(20261081)
  model=AutoModel.from_pretrained(m['model'],trust_remote_code=True,device='cuda:0',local_files_only=True)
  cases=read(Path(m['cases']));deadline=time.monotonic()+m['max_seconds']
  for i,case in enumerate(cases):
   for order in ['canonical']+(['reversed'] if i<4 else []):
    assert time.monotonic()<deadline
    req=copy.deepcopy(case['request'])
    if order=='reversed':req['questions']['next']['criteria']=dict(reversed(list(req['questions']['next']['criteria'].items())))
    torch.cuda.synchronize();t=time.monotonic();response=model.system_one(**req);torch.cuda.synchronize()
    choice=response['answers']['next']['choice'];assert choice in req['questions']['next']['criteria'],response
    pred.append(dict(qid=case['qid'],turn=case['turn'],order=order,response=response,seconds=time.monotonic()-t,request_sha256=hashlib.sha256(json.dumps(req,sort_keys=True,ensure_ascii=False).encode()).hexdigest()))
    save(out/'predictions.json',pred);phase('inference')
  save(out/'runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,peak_allocated_bytes=torch.cuda.max_memory_allocated(),controlled_latency=False,model_revision=m['revision']))
  phase('complete')
 except BaseException as e:phase('failed',error=repr(e));raise
if __name__=='__main__':main()
