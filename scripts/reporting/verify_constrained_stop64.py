"""Frozen development comparison; official scores and correctness-separated cost."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[2]
NAME='decision_constrained_stop72_validation64_20261024'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(es):
 n=len(es);g=sum(e['generated_tokens'] for e in es)
 return dict(questions=n,total_generated=g,mean_generated=g/n if n else None)
def main():
 run=ROOT/'results'/NAME;spec=read(run/'spec.json');status=read(run/'status.json');assert status['phase']=='complete'
 plan=read(REPO/'lab/registry/decision_constrained_stop72_20261024/plan.json')
 selection=read(REPO/'lab/registry/decision_shared_final_20261023/selection_plan.json');assert not set(spec['qids'])&set(selection['test_qids']+selection['reserved_qids']+plan['fit'])
 baseline=ROOT/'results/decision_shared_validation64_20261023';bs=read(baseline/'spec.json')
 for k in ['qids','model','corpus','document_sha256','vectors_sha256']:assert spec[k]==bs[k]
 for path,h in spec['source_sha256'].items():assert sha(REPO/path)==h
 for v in spec['policies'].values():assert sha(Path(v['checkpoint']))==v['sha256']
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1,compute_exact
 gold={}
 for split in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
   q=json.loads(line);gold[q['id']]=[q['answer']]+q['answer_aliases']
 rows={};hashes={}
 audit=read(REPO/'lab/reports/decision_h3_role_audit_20261024/audit.json');assert audit['verified']
 paths=list((run/'episodes').glob('*.json'));assert len(paths)==128==status['episodes']
 paths += [baseline/'episodes'/f'shared_grpo_{q}.json' for q in spec['qids']]
 for path in paths:
  e=read(path);assert e['qid'] in spec['qids'];assert e['f1']==max(compute_f1(a,e['answer']) for a in gold[e['qid']])
  assert e['generated_tokens']==sum(v.get('generated_tokens',0) for v in e['reader_calls'])
  e['em']=max(compute_exact(a,e['answer']) for a in gold[e['qid']]);assert (e['policy'],e['qid']) not in rows
  if e['policy']=='shared_grpo':assert sha(path)==audit['episode_sha256'][str(path)]
  rows[e['policy'],e['qid']]=e;hashes[str(path)]=sha(path)
 assert len(rows)==192 and sum(e['generated_tokens'] for (p,q),e in rows.items() if p!='shared_grpo')==status['generated']
 result={}
 for policy in ['h3','constrained_stop','shared_grpo']:
  es=[rows[policy,q] for q in spec['qids']];correct=[e for e in es if e['em']];incorrect=[e for e in es if not e['em']]
  result[policy]=dict(f1=sum(e['f1'] for e in es)/64,all=stats(es),correct=stats(correct),incorrect=stats(incorrect),partial=stats([e for e in incorrect if 0<e['f1']<1]))
 paired={}
 for ref in ['h3','shared_grpo']:
  groups={}
  for a,b,label in [(1,1,'both_correct'),(0,1,'candidate_only_correct'),(1,0,'reference_only_correct'),(0,0,'both_incorrect')]:
   ids=[q for q in spec['qids'] if rows[ref,q]['em']==a and rows['constrained_stop',q]['em']==b]
   old=stats([rows[ref,q] for q in ids]);new=stats([rows['constrained_stop',q] for q in ids]);groups[label]=dict(reference=old,candidate=new,saved_tokens=old['total_generated']-new['total_generated'])
  paired[ref]=groups
 for q in spec['qids']:
  old=read(ROOT/'results/decision_budget_stop_h3_validation64_20261024/episodes'/f'budget_stop_{q}.json')
  assert all(old[k]==rows['h3',q][k] for k in ['answer','f1','generated_tokens','prompt_tokens'])
 a,b=result['shared_grpo'],result['constrained_stop'];errors=sum(bool(e['errors']) for e in rows.values())
 screen=b['f1']>=a['f1']+.05 and b['all']['total_generated']<=.7*a['all']['total_generated']
 changes=[]
 for q in spec['qids']:
  h,c=rows['h3',q],rows['constrained_stop',q]
  if h['f1']!=c['f1'] or h['em']!=c['em']:changes.append(dict(qid=q,h3_f1=h['f1'],candidate_f1=c['f1'],h3_em=h['em'],candidate_em=c['em'],h3_actions=[d['action'] for d in h['decisions']],candidate_actions=[d['action'] for d in c['decisions']]))
 report=dict(verified=True,scope='Repeatedly exposed development64, not independent confirmation',policies=result,paired=paired,errors=errors,development_joint_screen=screen,goal_achieved=False,h3_reference_replay=64,quality_changes=changes,episode_sha256=hashes)
 out=REPO/'lab/reports'/NAME;out.mkdir(parents=True,exist_ok=True);(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='episode_sha256'},indent=2))
if __name__=='__main__':main()
