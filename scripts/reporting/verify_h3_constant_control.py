"""Rescore raw control episodes, verify token-only selection and archive bytes."""
import collections,hashlib,json,sys,tarfile
from pathlib import Path
import torch
from run_decision_policy_pilot import ROOT,REPO,sha,save

def read(p):return json.loads(p.read_text())
def main():
 name='decision_h3_constant_control_20261024';reg=REPO/'lab/registry'/name;plan=read(reg/'plan.json');selection=read(reg/'selection.json')
 assert read(ROOT/'results/decision_h3_constant_coordinator_20261024.json')['phase']=='complete'
 for p,h in plan['input_sha256'].items():assert sha(Path(p))==h
 for p,h in plan['source_sha256'].items():assert sha(REPO/p)==h
 raw={}
 for split in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
   q=json.loads(line);raw[q['id']]=q
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1
 h3=torch.load(ROOT/'results/decision_budget_stop_h3_20261024/stop.pt',weights_only=True)
 for c in plan['candidates']:
  spec=read(reg/'calibration_spec.json')['policies'][c['name']];assert sha(Path(spec['checkpoint']))==spec['sha256']
  state=torch.load(spec['checkpoint'],weights_only=True)
  assert torch.count_nonzero(state['stop.weight'])==0
  assert abs(float(state['stop.bias'].item())-c['bias'])<1e-6
  assert all(torch.equal(v,h3[k]) for k,v in state.items() if k.startswith('base.'))
 runs=['decision_h3_constant_calibration24_20261024','decision_h3_constant_validation64_20261024'];evidence={}
 for run in runs:
  p=ROOT/'results'/run;status=read(p/'status.json');spec=read(p/'spec.json');assert status['phase']=='complete'
  eps=[read(f) for f in (p/'episodes').glob('*.json')]
  assert len(eps)==status['episodes']==len(spec['policies'])*len(spec['qids'])
  for k,f in [('generated','generated_tokens'),('inputs','prompt_tokens')]:assert sum(e[f] for e in eps)==status[k]
  for policy in spec['policies']:
   for qid in spec['qids']:
    e=read(p/'episodes'/f'{policy}_{qid}.json');q=raw[qid]
    assert e['f1']==max(compute_f1(a,e['answer']) for a in [q['answer']]+q['answer_aliases'])
    assert e['generated_tokens']==sum(c.get('generated_tokens',0) for c in e['reader_calls'])
  archive=read(REPO/'lab/registry'/run/'archive.json');assert sha(Path(archive['path']))==archive['sha256']
  with tarfile.open(archive['path']) as tf:
   for member in tf.getmembers():
    if member.isfile():
     f=p.joinpath(*Path(member.name).parts[1:]);assert f.is_file()
     assert hashlib.sha256(tf.extractfile(member).read()).hexdigest()==sha(f)
  evidence[run]=dict(counts=status,archive_sha256=archive['sha256'],errors=sum(bool(e['errors']) for e in eps))
 cal=ROOT/'results'/runs[0];cs=read(cal/'spec.json');assert cs['qids']==plan['calibration_questions']
 assert set(cs['qids'])<=set(plan['training_questions']) and not set(cs['qids'])&set(plan['validation']+plan['test'])
 means={policy:sum(read(cal/'episodes'/f'{policy}_{qid}.json')['generated_tokens'] for qid in cs['qids'])/len(cs['qids']) for policy in cs['policies']}
 best=min(plan['candidates'],key=lambda c:(abs(means[c['name']]-means['h3_reference']),abs(c['bias']),c['name']))
 assert selection['selected']==best and selection['generated_means']==means and not selection['f1_used']
 dev=ROOT/'results'/runs[1];ds=read(dev/'spec.json');assert ds['qids']==plan['validation'];assert ds['policies']['constant_stop']==selection['checkpoint']
 groups=collections.defaultdict(lambda:dict(n=0,constant_minus_h3_tokens=0));exact=0
 for qid in ds['qids']:
  a=read(dev/'episodes'/f'selected_reference_{qid}.json');b=read(dev/'episodes'/f'constant_stop_{qid}.json');old=read(ROOT/'results/decision_budget_stop_h3_validation64_20261024/episodes'/f'budget_stop_{qid}.json')
  assert all(a[k]==old[k] for k in ['answer','f1','generated_tokens','prompt_tokens']);exact+=1
  group='constant_better' if b['f1']>a['f1'] else 'h3_better' if b['f1']<a['f1'] else 'positive_tie' if a['f1']>0 else 'both_zero'
  groups[group]['n']+=1;groups[group]['constant_minus_h3_tokens']+=b['generated_tokens']-a['generated_tokens']
 summary=read(dev/'summary.json')['policies'];ratio=summary['constant_stop']['generated_tokens']/summary['selected_reference']['generated_tokens']
 save(REPO/'lab/reports'/runs[1]/'independent_verification.json',dict(verified=True,reference_exact=exact,selection=selection,dev_cost_ratio=ratio,dev_cost_match_pass=abs(ratio-1)<=plan['cost_match_tolerance'],groups=dict(groups),runs=evidence))
 print(json.dumps(dict(summary=summary,train_cost_ratio=selection['cost_ratio'],dev_cost_ratio=ratio,groups=dict(groups)),indent=2))
if __name__=='__main__':main()
