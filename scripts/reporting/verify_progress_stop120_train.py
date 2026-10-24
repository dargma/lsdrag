"""Raw full-episode quality gate with exact-correct/incorrect cost separation."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[2]
NAME='decision_progress_stop120_train_eval_20261024'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(es):
 n=len(es);g=sum(e['generated_tokens'] for e in es)
 return dict(questions=n,total_generated=g,mean_generated=g/n if n else None)
def main():
 run=ROOT/'results'/NAME;spec=read(run/'spec.json');status=read(run/'status.json');assert status['phase']=='complete'
 plan=read(REPO/'lab/registry/decision_progress_stop120_20261024/plan.json');assert spec['qids']==plan['fit']
 selection=read(REPO/'lab/registry/decision_shared_final_20261023/selection_plan.json');assert not set(spec['qids'])&set(selection['test_qids']+selection['reserved_qids']+plan['excluded'])
 for path,h in spec['source_sha256'].items():assert sha(REPO/path)==h
 for v in spec['policies'].values():assert sha(Path(v['checkpoint']))==v['sha256']
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1,compute_exact
 gold={}
 for split in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
   q=json.loads(line);gold[q['id']]=[q['answer']]+q['answer_aliases']
 rows={};hashes={}
 for path in (run/'episodes').glob('*.json'):
  e=read(path);assert e['qid'] in spec['qids'];assert e['f1']==max(compute_f1(a,e['answer']) for a in gold[e['qid']])
  assert e['generated_tokens']==sum(v.get('generated_tokens',0) for v in e['reader_calls'])
  e['em']=max(compute_exact(a,e['answer']) for a in gold[e['qid']]);assert (e['policy'],e['qid']) not in rows
  rows[e['policy'],e['qid']]=e;hashes[str(path)]=sha(path)
 assert len(rows)==240==status['episodes']
 assert sum(e['generated_tokens'] for e in rows.values())==status['generated']
 result={}
 for policy in ['h3','progress_stop']:
  es=[rows[policy,q] for q in spec['qids']];correct=[e for e in es if e['em']];incorrect=[e for e in es if not e['em']]
  result[policy]=dict(f1=sum(e['f1'] for e in es)/120,all=stats(es),correct=stats(correct),incorrect=stats(incorrect),partial=stats([e for e in incorrect if 0<e['f1']<1]))
 paired={}
 for a,b,label in [(1,1,'both_correct'),(0,1,'candidate_only_correct'),(1,0,'h3_only_correct'),(0,0,'both_incorrect')]:
  ids=[q for q in spec['qids'] if rows['h3',q]['em']==a and rows['progress_stop',q]['em']==b]
  ref=stats([rows['h3',q] for q in ids]);cand=stats([rows['progress_stop',q] for q in ids]);paired[label]=dict(h3=ref,candidate=cand,saved_tokens=ref['total_generated']-cand['total_generated'])
 for q in spec['qids']:
  oldpath=ROOT/'results/decision_shared_residual72_train_eval_20261024/episodes'/f'h3_{q}.json'
  if not oldpath.exists():oldpath=ROOT/'results/decision_h3_stop_coverage48_20261024/episodes'/f'{q}_collector.json'
  old=read(oldpath)
  assert all(old[k]==rows['h3',q][k] for k in ['answer','f1','generated_tokens','prompt_tokens'])
 for q in spec['qids']:
  h,c=rows['h3',q]['decisions'][0],rows['progress_stop',q]['decisions'][0]
  assert h['action']==c['action'] and h['policy_probs']==c['policy_probs'],'First decision must match H3'
 a,b=result['h3'],result['progress_stop'];errors=sum(bool(e['errors']) for e in rows.values())
 weak=b['f1']>=a['f1'] and b['correct']['questions']>=a['correct']['questions'] and b['all']['total_generated']<=a['all']['total_generated']
 strict=b['f1']>a['f1'] or b['correct']['questions']>a['correct']['questions'] or b['all']['total_generated']<a['all']['total_generated']
 useful=paired['both_correct']['saved_tokens']>0 or paired['candidate_only_correct']['saved_tokens']>0
 passed=weak and strict and useful and paired['h3_only_correct']['h3']['questions']==0 and errors==0
 report=dict(verified=True,scope='Training120 full rollouts; not generalization or goal achievement',policies=result,paired=paired,errors=errors,train_gate=passed,h3_reference_replay=120,initial_h3_exact=120,episode_sha256=hashes)
 out=REPO/'lab/reports'/NAME;out.mkdir(parents=True,exist_ok=True);(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='episode_sha256'},indent=2))
if __name__=='__main__':main()
