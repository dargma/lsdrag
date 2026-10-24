"""Independent raw terminal-outcome audit for the current-H3 all-action collection."""
import json,sys,tarfile,hashlib
from pathlib import Path
import torch
from run_decision_policy_pilot import ROOT,REPO,sha,save
NAME='decision_h3_stop_coverage48_20261024'
def read(p):return json.loads(p.read_text())
def main():
 p=ROOT/'results'/NAME;m=read(p/'manifest.json');status=read(p/'status.json');assert status['phase']=='complete'
 for f,h in m['source_sha256'].items():assert sha(REPO/f)==h
 for f,h in m['input_sha256'].items():assert sha(Path(f))==h
 records=torch.load(p/'states.pt',weights_only=True);eps=[read(f) for f in (p/'episodes').glob('*.json')];by={(e['qid'],e['tag']):e for e in eps}
 expected={};expected_branches=0
 for qid in m['train']:
  parent=read(p/'episodes'/f'{qid}_collector.json');prefills=[c for c in parent['reader_calls'] if c.get('pack_slots')]
  turns=[i for i,d in enumerate(parent['decisions']) if not d['forced']]
  for i in turns:expected[qid,i]=(parent['decisions'][i],prefills[i]['messages_sha256'],parent);expected_branches+=2
 assert len(records)==len(expected)==status['states']
 assert {(r['qid'],r['turn']) for r in records}==set(expected)
 assert len(eps)==len(by)==expected_branches+len(m['train'])==status['episodes']
 assert set(m['train'])<=set(m['train_pool']) and not set(m['train'])&set(m['validation']+m['test'])
 raw={}
 for split in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
   q=json.loads(line)
   if q['id'] in m['train']:raw[q['id']]=q
 assert set(raw)==set(m['train']) and not set(m['train'])&set(m['prior_outcome_questions'])
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1,compute_exact
 for qid in m['train']:
  ep=read(p/'episodes'/f'{qid}_collector.json');q=raw[qid]
  assert ep['f1']==max(compute_f1(a,ep['answer']) for a in [q['answer']]+q['answer_aliases'])
 for r in records:
  d,prefix,parent=expected[r['qid'],r['turn']];assert prefix==r['prefix_sha256']
  assert r['reference_probs']==d['policy_probs'] and r['legal'].tolist()==d['legal']
  assert r['pack'].dtype==torch.float32 and torch.isfinite(r['pack']).all()
  assert {s['action'] for s in r['samples']}=={3,r['continue_action']}
  assert r['continue_action']==max((a for a in range(3) if r['legal'][a]),key=lambda a:r['reference_probs'][a])
  assert len(r['samples'])==2
  for s in r['samples']:
   assert s['rep']==0;e=by[r['qid'],f"t{r['turn']}_a{s['action']}_r0"];q=raw[r['qid']]
   assert e['f1']==s['f1']==max(compute_f1(a,e['answer']) for a in [q['answer']]+q['answer_aliases'])
   assert e['reader_calls'][0]['messages_sha256']==prefix and e['initial_tool_calls']==r['turn']
   assert e['decisions'][0]['action']==s['action'] and e['decisions'][0]['branch_intervention']
   for v in e['decisions'][1:]:assert v['action']==max(range(4),key=v['policy_probs'].__getitem__) and not v['branch_intervention']
   assert e['errors']==s['errors'] and e['generated_tokens']==s['generated'] and e['prompt_tokens']==s['inputs']
   assert e['generated_tokens']==sum(v.get('generated_tokens',0) for v in e['reader_calls'])
   if s['action']==d['action']:
    assert e['answer']==parent['answer'] and e['f1']==parent['f1'],'Reference suffix did not replay'
 for key,field in [('generated','generated_tokens'),('inputs','prompt_tokens')]:assert status[key]==sum(e[field] for e in eps)
 archive=read(REPO/'lab/registry'/NAME/'archive.json');assert sha(Path(archive['path']))==archive['sha256']
 with tarfile.open(archive['path']) as tf:
  for member in tf.getmembers():
   if member.isfile():
    local=p.joinpath(*Path(member.name).parts[1:]);assert local.is_file()
    assert hashlib.sha256(tf.extractfile(member).read()).hexdigest()==sha(local)
 correct=[];incorrect=[];partial=[]
 for e in eps:
  q=raw[e['qid']];em=max(compute_exact(a,e['answer']) for a in [q['answer']]+q['answer_aliases'])
  (correct if em else incorrect).append(e)
  if not em and 0<e['f1']<1:partial.append(e)
 def cost(es):return dict(episodes=len(es),total_generated=sum(e['generated_tokens'] for e in es),mean_generated=sum(e['generated_tokens'] for e in es)/len(es) if es else None)
 report=dict(correctness_cost=dict(correct=cost(correct),incorrect=cost(incorrect),partial=cost(partial),scope='Collection workload including overlapping suffixes, not policy performance'),verified=True,questions=len(raw),states=len(records),episodes=len(eps),errors=sum(bool(e['errors']) for e in eps),counts=status,states_sha256=sha(p/'states.pt'),archive_sha256=archive['sha256'],note='Terminal F1 and suffix-generation evidence under current H3; not improved-policy quality.')
 save(REPO/'lab/reports'/NAME/'independent_verification.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
