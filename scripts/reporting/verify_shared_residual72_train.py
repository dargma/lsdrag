"""Raw paired-training rollout gate, including frozen H3 reference replay."""
import json,sys,hashlib
from pathlib import Path
from collections import defaultdict
ROOT=Path('/content/local_fast/lsdrag-text-v1/results');REPO=Path('/content/local_fast/lsdrag-code')
NAME='decision_shared_residual72_train_eval_20261024'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 run=ROOT/NAME;spec=read(run/'spec.json');status=read(run/'status.json');assert status['phase']=='complete'
 for f,h in spec['source_sha256'].items():assert sha(REPO/f)==h
 for desc in spec['policies'].values():assert sha(Path(desc['checkpoint']))==desc['sha256']
 plan=read(REPO/'lab/registry/decision_shared_residual72_20261024/plan.json');assert spec['qids']==plan['fit']
 assert not set(spec['qids'])&set(plan['excluded'])
 raw={}
 for split in ['train','dev']:
  for line in (ROOT.parent/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
   q=json.loads(line)
   if q['id'] in spec['qids']:raw[q['id']]=q
 sys.path.insert(0,str(ROOT.parent/'source/musique_scorer'));from metrics.answer import compute_f1
 eps=[read(p) for p in (run/'episodes').glob('*.json')];by={(e['qid'],e['policy']):e for e in eps};assert len(eps)==len(by)==144
 references={}
 for dirname,pattern in [('decision_h3_constant_calibration24_20261024','h3_reference_*.json'),('decision_h3_policy_coverage24b_20261024','*_collector.json'),('decision_h3_diversity48_20261024','*_collector.json')]:
  for p in (ROOT/dirname/'episodes').glob(pattern):
   e=read(p);references[e['qid']]=e
 assert set(spec['qids'])<=set(references)
 for e in eps:
  q=raw[e['qid']];assert e['f1']==max(compute_f1(a,e['answer']) for a in [q['answer']]+q['answer_aliases'])
  assert e['generated_tokens']==sum(c.get('generated_tokens',0) for c in e['reader_calls'])
  for d in e['decisions']:assert d['legal'][d['action']]
  if e['policy']=='h3':
   ref=references[e['qid']]
   for k in ['answer','f1','generated_tokens','tool_calls']:assert e[k]==ref[k],(e['qid'],k)
 summary=read(run/'summary.json');computed={}
 for name in spec['policies']:
  rows=[e for e in eps if e['policy']==name]
  computed[name]={k:sum(e[k] for e in rows)/len(rows) for k in ['f1','generated_tokens','prompt_tokens','tool_calls']}
  for k,v in computed[name].items():assert abs(v-summary['policies'][name][k])<1e-8
 for field,key in [('generated_tokens','generated'),('prompt_tokens','inputs')]:assert sum(e[field] for e in eps)==status[key]
 savings=defaultdict(int);useful=[];wins=losses=0
 for qid in spec['qids']:
  a=by[qid,'h3'];b=by[qid,'shared_residual'];delta=b['f1']-a['f1'];saved=a['generated_tokens']-b['generated_tokens']
  bucket='quality_gain' if delta>0 else 'quality_loss' if delta<0 else 'positive_retained' if a['f1']>0 else 'both_zero';savings[bucket]+=saved
  wins+=delta>0;losses+=delta<0
  if delta>0 or (delta==0 and a['f1']>0 and saved>0):useful.append(qid)
 a=computed['h3'];b=computed['shared_residual'];gate=b['f1']>=a['f1'] and b['generated_tokens']<=a['generated_tokens'] and bool(useful)
 report=dict(verified=True,policies=computed,train_gate_passed=gate,quality_wins=wins,quality_losses=losses,savings=dict(savings),useful_questions=useful,reference_replay=72,errors=sum(bool(e['errors']) for e in eps),scope='Training policy quality only; not development or independent confirmation')
 dest=REPO/'lab/reports'/NAME;dest.mkdir(parents=True,exist_ok=True);(dest/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
