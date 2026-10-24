"""Same-question development comparison to immutable GRPO, raw rescoring and useful cost."""
import json,sys,hashlib,tarfile
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path('/content/local_fast/lsdrag-text-v1/results');REPO=Path('/content/local_fast/lsdrag-code');NAME='decision_shared_residual72_validation64_20261024'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=ROOT/NAME;spec=read(p/'spec.json');status=read(p/'status.json');assert status['phase']=='complete'
 base=ROOT/'decision_shared_validation64_20261023';bs=read(base/'spec.json')
 assert bs['policies']['shared_grpo']['sha256']=='59a0f76672a3411abaa53ab8fd8f40e0c8ddc027d26d566fea25a7187e4421ef'
 for key in ['qids','model','corpus','document_sha256','vectors_sha256']:assert spec[key]==bs[key],key
 for f,h in spec['source_sha256'].items():assert sha(REPO/f)==h
 for d in spec['policies'].values():assert sha(Path(d['checkpoint']))==d['sha256']
 selection=read(REPO/'lab/registry/decision_shared_final_20261023/selection_plan.json');assert not set(spec['qids'])&set(selection['test_qids']+selection['reserved_qids'])
 plan=read(REPO/'lab/registry/decision_shared_residual72_20261024/plan.json');assert not set(spec['qids'])&set(plan['fit'])
 gold={}
 for split in ['train','dev']:
  for line in (ROOT.parent/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
   q=json.loads(line)
   if q['id'] in spec['qids']:gold[q['id']]=q
 sys.path.insert(0,str(ROOT.parent/'source/musique_scorer'));from metrics.answer import compute_f1
 eps=[read(f) for f in (p/'episodes').glob('*.json')];assert len(eps)==128==status['episodes'];by={(e['policy'],e['qid']):e for e in eps};assert len(by)==128
 rows={'basic_grpo':[],'h3':[],'shared_residual':[]};groups=defaultdict(lambda:dict(questions=0,saved_tokens=0));changes=[]
 for qid in spec['qids']:
  a=by['h3',qid];b=by['shared_residual',qid];g=read(base/'episodes'/f'shared_grpo_{qid}.json');old=read(ROOT/'decision_budget_stop_h3_validation64_20261024/episodes'/f'budget_stop_{qid}.json')
  assert all(a[k]==old[k] for k in ['answer','f1','generated_tokens','prompt_tokens'])
  for label,e in [('h3',a),('shared_residual',b),('basic_grpo',g)]:
   q=gold[qid];assert e['f1']==max(compute_f1(answer,e['answer']) for answer in [q['answer']]+q['answer_aliases'])
   assert e['generated_tokens']==sum(c.get('generated_tokens',0) for c in e['reader_calls']);rows[label].append(e)
  group='quality_gain' if b['f1']>a['f1'] else 'quality_loss' if b['f1']<a['f1'] else 'positive_retained' if a['f1']>0 else 'both_zero'
  groups[group]['questions']+=1;groups[group]['saved_tokens']+=a['generated_tokens']-b['generated_tokens']
  if a['f1']!=b['f1']:changes.append(dict(qid=qid,h3_f1=a['f1'],candidate_f1=b['f1'],h3_actions=[d['action'] for d in a['decisions']],candidate_actions=[d['action'] for d in b['decisions']]))
 for k,field in [('generated','generated_tokens'),('inputs','prompt_tokens')]:assert status[k]==sum(e[field] for e in eps)
 result={name:{k:float(np.mean([e[k] for e in es])) for k in ['f1','generated_tokens','prompt_tokens','tool_calls']} for name,es in rows.items()}
 g=result['basic_grpo'];c=result['shared_residual'];deltas={};rng=np.random.default_rng(20261080);indices=rng.integers(0,64,(10000,64))
 for ref in ['h3','basic_grpo']:
  dif=np.array([b['f1']-a['f1'] for a,b in zip(rows[ref],rows['shared_residual'])]);deltas[ref]={'f1_delta':float(dif.mean()),'f1_paired_bootstrap95':np.quantile(dif[indices].mean(1),[.025,.975]).tolist(),'generated_ratio':c['generated_tokens']/result[ref]['generated_tokens']}
 summary=read(p/'summary.json')
 for name in ['h3','shared_residual']:
  for k,v in result[name].items():assert abs(v-summary['policies'][name][k])<1e-8
 archive=read(REPO/'lab/registry'/NAME/'archive.json');assert sha(Path(archive['path']))==archive['sha256']
 with tarfile.open(archive['path']) as tf:
  for member in tf.getmembers():
   if member.isfile():assert hashlib.sha256(tf.extractfile(member).read()).hexdigest()==sha(p.joinpath(*Path(member.name).parts[1:]))
 report=dict(verified=True,policies=result,deltas=deltas,groups=dict(groups),quality_changes=changes,goal_screen=c['f1']>=g['f1']+.05 and c['generated_tokens']<=g['generated_tokens']*.7,reference_replay=64,errors=sum(bool(e['errors']) for e in eps),archive_sha256=archive['sha256'],scope='Repeatedly exposed development64; no independent confirmation or goal completion')
 dest=REPO/'lab/reports'/NAME;dest.mkdir(parents=True,exist_ok=True);(dest/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
