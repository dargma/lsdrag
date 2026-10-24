import json,hashlib
from pathlib import Path
import torch
from lsdrag.text_integrated import canonical
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path('/content/local_fast/lsdrag-code');NAME='decision_d3_nano_local32_20261024'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 plan=read(REPO/'lab/registry/decision_future_critic_probe72_20261024/plan.json');records=[]
 for path in plan['inputs']:records.extend(torch.load(path,map_location='cpu',weights_only=True))
 qids=sorted(plan['fit'],key=lambda q:hashlib.sha256(('20261081:'+q).encode()).hexdigest())[:32]
 parents={}
 for name,pattern in [('decision_h3_constant_calibration24_20261024','h3_reference_*.json'),('decision_h3_policy_coverage24b_20261024','*_collector.json'),('decision_h3_diversity48_20261024','*_collector.json')]:
  for p in (ROOT/'results'/name/'episodes').glob(pattern):
   e=read(p);parents[e['qid']]=(p,e)
 raw={}
 for split in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
   q=json.loads(line)
   if q['id'] in qids:raw[q['id']]=q
 prompt=(REPO/'prompts/text_rag_integrated_v5.txt').read_text();cases=[];inputs={p:sha(Path(p)) for p in plan['inputs']}
 descriptions={'keyword_search':'Search the corpus using exact keywords to find relevant evidence.','semantic_search':'Search the corpus by meaning to find relevant evidence.','read_chunk':'Read a previously located document chunk in more detail.','FINAL':'Stop using tools and generate the final answer from available evidence.'}
 actions=list(descriptions)
 for qid in qids:
  r=min([r for r in records if r['qid']==qid],key=lambda r:hashlib.sha256((qid+':'+str(r['turn'])).encode()).hexdigest());p,parent=parents[qid];inputs[str(p)]=sha(p)
  messages=[{'role':'system','content':prompt},{'role':'user','content':raw[qid]['question']}]
  for i,step in enumerate(parent['trajectory'][:r['turn']]):
   tc={'id':f'call_{i}','type':'function','function':{'name':step['tool_name'],'arguments':canonical(step['arguments'])}}
   messages.extend([{'role':'assistant','content':None,'tool_calls':[tc]},{'role':'tool','tool_call_id':tc['id'],'content':canonical(step['result'])}])
  assert hashlib.sha256(canonical(messages).encode()).hexdigest()==r['prefix_sha256']
  req={'state':canonical(messages),'questions':{'next':{'type':'choice','instructions':'Choose the next action to maximize final answer correctness while avoiding unnecessary tool calls. Continue gathering evidence if needed; choose FINAL when the available evidence is sufficient.','criteria':{name:descriptions[name] for i,name in enumerate(actions) if r['legal'][i]}}}}
  cases.append(dict(qid=qid,turn=r['turn'],prefix_sha256=r['prefix_sha256'],request=req,reference_action=int(torch.tensor(r['reference_probs']).argmax()),samples=r['samples']))
 out=ROOT/'results'/NAME;out.mkdir(exist_ok=False);(out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
 snapshot=ROOT/'hf/models--vllm-sr--d3-nano/snapshots/462efc366229ea268cf4eed3738b3fc4415e4817'
 modelhash={str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()}
 manifest=dict(model=str(snapshot),revision=snapshot.name,model_sha256=modelhash,cases=str(out/'cases.json'),cases_sha256=sha(out/'cases.json'),input_sha256=inputs,qids=qids,source_sha256={f:sha(REPO/f) for f in ['scripts/prepare_d3_nano_local32.py','docs/JEV_D3_NANO_LOCAL32_20261024.md']},max_seconds=1800,scope='training32 one-intervention H3-suffix outcome lookup; not fullpolicy eval')
 reg=REPO/'lab/registry'/NAME;reg.mkdir(parents=True,exist_ok=True);(reg/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Frozen32 predecision cases, exactprefixhashes, model bytes verified')
if __name__=='__main__':main()
