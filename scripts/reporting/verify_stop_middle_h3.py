"""Independently rescore complete middle-state collection and test H3 coverage."""
import json, sys, tarfile, hashlib
from pathlib import Path
import torch
from run_decision_policy_pilot import ROOT, REPO, save, sha
from lsdrag.decision_budget_targets import budget_targets

NAME='decision_stop_middle_h3_20261024'

def main():
 p=ROOT/'results'/NAME
 read=lambda p:json.loads(p.read_text())
 m=read(p/'manifest.json');status=read(p/'status.json');assert status['phase']=='complete'
 for path,h in m['input_sha256'].items():assert sha(Path(path))==h
 for path,h in m['source_sha256'].items():assert sha(REPO/path)==h
 records=torch.load(p/'states.pt',weights_only=True)
 parent=Path(m['parent_run']);old=torch.load(parent/'states.pt',weights_only=True)
 expected={}
 for qid in m['train']:
  ep=read(parent/'episodes'/f'{qid}_collector.json');valid=[i for i,d in enumerate(ep['decisions']) if not d['forced']]
  prefills=[u for u in ep['reader_calls'] if u.get('pack_slots')]
  for i in valid[1:-1]:expected[qid,i]=(ep['decisions'][i],prefills[i]['messages_sha256'])
 assert {(r['qid'],r['turn']) for r in records}==set(expected)
 assert len(records)==len(expected)==m['expected_states']==status['states']
 raw={}
 for split in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
   q=json.loads(line)
   if q['id'] in m['train']:raw[q['id']]=q
 assert set(raw)==set(m['train'])<=set(m['train_pool'])
 assert not set(raw)&set(m['validation']+m['test'])
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1
 eps=[read(f) for f in (p/'episodes').glob('*.json')];by={(e['qid'],e['tag']):e for e in eps}
 assert len(eps)==len(by)==2*len(records)==m['expected_episodes']==status['episodes']
 for r in records:
  d,prefix=expected[r['qid'],r['turn']];assert prefix==r['prefix_sha256']
  assert r['reference_probs']==d['policy_probs'] and r['legal'].tolist()==d['legal']
  scores=torch.tensor(d['policy_probs'][:3]).masked_fill(~r['legal'][:3],-1.)
  assert int(scores.argmax())==r['continue_action'];assert r['pack'].dtype==torch.float32
  assert {s['action'] for s in r['samples']}=={3,r['continue_action']} and len(r['samples'])==2
  for s in r['samples']:
   e=by[r['qid'],f"t{r['turn']}_a{s['action']}_r0"];q=raw[r['qid']]
   assert e['f1']==s['f1']==max(compute_f1(a,e['answer']) for a in [q['answer']]+q['answer_aliases'])
   assert e['reader_calls'][0]['messages_sha256']==prefix
   assert e['decisions'][0]['action']==s['action'] and e['decisions'][0]['branch_intervention']
   assert e['errors']==s['errors'] and e['generated_tokens']==s['generated'] and e['prompt_tokens']==s['inputs']
 for key,field in [('generated','generated_tokens'),('inputs','prompt_tokens')]:assert status[key]==sum(e[field] for e in eps)
 archive=read(REPO/'lab/registry'/NAME/'archive.json')
 print('archive fields',list(archive))
 # Verify archive bytes against every runtime file, without extracting.
 archive_path=Path(archive.get('archive',archive.get('path','')))
 assert archive_path.is_file(),archive
 assert sha(archive_path)==archive['sha256']
 with tarfile.open(archive_path) as tf:
  files={x.name:x for x in tf.getmembers() if x.isfile()}
  for f in p.rglob('*'):
   if f.is_file():
    matches=[x for x in files if x.endswith('/'+str(f.relative_to(p)))]
    assert len(matches)==1
    assert hashlib.sha256(tf.extractfile(files[matches[0]]).read()).hexdigest()==sha(f)
 combined=old+records
 assert len({(r['qid'],r['prefix_sha256']) for r in combined})==len(combined)
 labels,evidence=budget_targets(combined)
 dest=REPO/'lab/reports'/NAME
 save(dest/'verification.json',dict(verified=True,states=len(records),questions=len(raw),episodes=len(eps),errors=sum(bool(e['errors']) for e in eps),states_sha256=sha(p/'states.pt'),counts=status,budget_feasibility=evidence,gate_passed=evidence['changed_questions']>=3))
 print(json.dumps(evidence,indent=2))
 if evidence['changed_questions']>=3:
  # Separate derived artifact, not a mutation of the archived collection.
  out=ROOT/'results/decision_budget_stop_h3_inputs_20261024';out.mkdir(exist_ok=False)
  torch.save(combined,out/'states.pt');save(out/'provenance.json',dict(parents={str(parent/'states.pt'):sha(parent/'states.pt'),str(p/'states.pt'):sha(p/'states.pt')},evidence=evidence,labels=labels))
if __name__=='__main__':main()
