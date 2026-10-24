"""Independent terminal QA/checkpoint/archive audit of the first policy update."""
import collections,hashlib,json,sys,tarfile
from pathlib import Path
import torch
from run_decision_policy_pilot import ROOT,REPO,save,sha
NAME='decision_h3_policy_iteration1_validation64_20261024'
def read(p):return json.loads(p.read_text())
def main():
 p=ROOT/'results'/NAME;status=read(p/'status.json');assert status['phase']=='complete'
 assert read(ROOT/'results/decision_h3_policy_iteration1_coordinator_20261024.json')['phase']=='complete'
 spec=read(p/'spec.json');plan=read(REPO/'lab/registry/decision_h3_policy_iteration1_20261024/plan.json')
 for f,h in spec['source_sha256'].items():assert sha(REPO/f)==h
 for f,h in plan['input_sha256'].items():assert sha(Path(f))==h
 assert spec['qids']==plan['validation'] and not set(spec['qids'])&set(plan['train_pool']+plan['test'])
 refdesc=spec['policies']['selected_reference'];cand=spec['policies']['policy_iteration1']
 for desc in [refdesc,cand]:assert sha(Path(desc['checkpoint']))==desc['sha256']
 before=torch.load(refdesc['checkpoint'],weights_only=True);after=torch.load(cand['checkpoint'],weights_only=True)
 changed=[k for k in before if not torch.equal(before[k],after[k])];assert set(changed)=={'base.net.2.weight','base.net.2.bias'}
 raw={}
 for split in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
   q=json.loads(line)
   if q['id'] in spec['qids']:raw[q['id']]=q
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1
 eps=[read(f) for f in (p/'episodes').glob('*.json')];assert len(eps)==status['episodes']==2*len(spec['qids'])
 for key,field in [('generated','generated_tokens'),('inputs','prompt_tokens')]:assert sum(e[field] for e in eps)==status[key]
 groups=collections.defaultdict(lambda:dict(n=0,candidate_minus_h3_tokens=0));examples=[];exact=0
 for qid in spec['qids']:
  a=read(p/'episodes'/f'selected_reference_{qid}.json');b=read(p/'episodes'/f'policy_iteration1_{qid}.json');old=read(ROOT/'results/decision_budget_stop_h3_validation64_20261024/episodes'/f'budget_stop_{qid}.json')
  assert all(a[k]==old[k] for k in ['answer','f1','generated_tokens','prompt_tokens']);exact+=1
  for e in [a,b]:
   q=raw[qid];assert e['f1']==max(compute_f1(v,e['answer']) for v in [q['answer']]+q['answer_aliases'])
   assert e['generated_tokens']==sum(c.get('generated_tokens',0) for c in e['reader_calls'])
  group='quality_gain' if b['f1']>a['f1'] else 'quality_loss' if b['f1']<a['f1'] else 'positive_retained' if a['f1']>0 else 'both_zero'
  groups[group]['n']+=1;groups[group]['candidate_minus_h3_tokens']+=b['generated_tokens']-a['generated_tokens']
  if group in ['quality_gain','quality_loss']:
   examples.append(dict(qid=qid,group=group,h3_f1=a['f1'],candidate_f1=b['f1'],token_delta=b['generated_tokens']-a['generated_tokens'],h3_actions=[d['action'] for d in a['decisions']],candidate_actions=[d['action'] for d in b['decisions']]))
 archive=read(REPO/'lab/registry'/NAME/'archive.json');assert sha(Path(archive['path']))==archive['sha256']
 with tarfile.open(archive['path']) as tf:
  for member in tf.getmembers():
   if member.isfile():
    f=p.joinpath(*Path(member.name).parts[1:]);assert f.is_file();assert hashlib.sha256(tf.extractfile(member).read()).hexdigest()==sha(f)
 report=dict(verified=True,reference_exact=exact,changed_checkpoint_keys=changed,groups=dict(groups),quality_changes=examples,counts=status,errors=sum(bool(e['errors']) for e in eps),archive_sha256=archive['sha256'])
 save(REPO/'lab/reports'/NAME/'independent_verification.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
