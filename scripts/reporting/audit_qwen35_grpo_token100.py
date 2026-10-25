"""Independent saved-trajectory reward/token/behavior-probability audit."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
from jsonschema import validate
from lsdrag.text_integrated import argument_schema,ACTIONS
from lsdrag.text_decision_benchmark import FINAL_SCHEMA
from lsdrag.decision_policy import distribution
from fit_qwen35_outcome_heads import base_load
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[2];NAME='qwen35_grpo_token100_20261025'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(2);run=ROOT/'results'/NAME;p=read(run/'plan.json');s=read(run/'status.json');summary=read(run/'summary.json');assert s['phase']=='complete' and s['episodes']==192
 for f,h in p['hashes'].items():assert sha(f)==h
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1,compute_exact
 protected=read(REPO/'lab/registry/decision_shared_final_20261023/selection_plan.json');splits=read(REPO/'lab/registry/qwen35_9b_h3_20261024/training_plan.json');assert p['qids']==splits['outcome_train48'];assert not set(p['qids'])&set(splits['development64']+protected['reserved_qids']+protected['test_qids'])
 raw={q['id']:q for q in map(json.loads,(ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open()) if q['id'] in p['qids']}
 eps={};hashes={};tokens=inputs=0
 for path in sorted((run/'episodes').glob('*.json')):
  e=read(path);assert e['qid'] in p['qids'];gold=[raw[e['qid']]['answer']]+raw[e['qid']]['answer_aliases'];em=max(compute_exact(a,e['answer']) for a in gold);f1=max(compute_f1(a,e['answer']) for a in gold);assert em==e['em'] and abs(f1-e['f1'])<1e-10
  assert e['generated_tokens']==sum(u.get('generated_tokens',0) for u in e['reader_calls'])
  for u in e['reader_calls']:
   if 'token_ids' in u:assert len(u['token_ids'])==u['generated_tokens']
  calls=[u for u in e['reader_calls'] if 'raw_json' in u];valid=not e['errors'] and len(calls)==len(e['decisions']) and e['decisions'][-1]['action']==3
  history=[]
  try:
   for d,u in zip(e['decisions'],calls):
    name=ACTIONS[d['action']];obj=json.loads(u['raw_json']);validate(obj,FINAL_SCHEMA if name=='FINAL' else argument_schema(name,history));valid=valid and u['finish_reason']=='schema_complete'
    if name!='FINAL':history.append(e['trajectory'][len(history)])
   valid=valid and obj=={'answer':e['answer'],'support_ids':e['support_ids']}
  except Exception:valid=False
  assert bool(valid)==e['format_valid'];expected=em+.25*f1-10.0*e['generated_tokens']/(e['generated_tokens']+256)+.02*valid;assert abs(e['reward']-expected)<1e-12
  key=(e['qid'],e['policy']);assert key not in eps;eps[key]=e;hashes[str(path)]=sha(path);tokens+=e['generated_tokens'];inputs+=e['prompt_tokens']
 assert len(eps)==192 and tokens==s['generated'] and inputs==s['inputs']
 groups=read(run/'groups.json')
 for q in p['qids']:
  rewards=np.array([eps[q,f'train{i}']['reward'] for i in range(4)]);assert np.allclose(rewards,groups[q]['rewards'],atol=1e-12);a=(rewards-rewards.mean())/(rewards.std()+1e-6);assert np.allclose(a,groups[q]['advantages'],atol=1e-12)
 batch=torch.load(run/'batch.pt',weights_only=True,map_location='cpu');assert len(batch)==summary['states'];anchor=base_load(p['checkpoint']);after=base_load(run/'head.pt')
 for b in batch:assert b['pack'].dtype==torch.float32 and not b['forced'] and b['reward']==eps[b['qid'],f"train{b['rep']}"]['reward']
 x=torch.stack([b['pack'] for b in batch]);m=torch.stack([b['legal'] for b in batch]);acts=torch.tensor([b['action'] for b in batch]);old=torch.tensor([b['old_logp'] for b in batch])
 with torch.no_grad():lp=distribution(anchor(x,m)[0],m).log_prob(acts)
 assert torch.allclose(lp,old,atol=2e-4,rtol=2e-4);changed=[k for k,v in after.state_dict().items() if not torch.equal(v,anchor.state_dict()[k])];assert changed==summary['changed_keys'];assert all(k.startswith('net.') for k in changed);assert all(torch.isfinite(v).all() for v in after.state_dict().values());assert sha(run/'head.pt')==summary['checkpoint_sha256']
 report=dict(verified=True,episodes=192,states=len(batch),raw_sha256=hashes,cpu_replay_max_error=float((lp-old).abs().max()),changed_keys=changed,format_group_variation=summary['format_group_variation'],reward_group_variation=summary['reward_group_variation'],scope='Full48q192trajectory training integrity; independent development evaluation required',summary=summary)
 out=REPO/'lab/reports'/NAME;out.mkdir(parents=True,exist_ok=True);(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['raw_sha256','summary']},indent=2))
if __name__=='__main__':main()
