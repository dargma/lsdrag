"""Raw same-first-action comparison for scripted search/read paths."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[2];NAME='decision_two_action12_20261024'
def read(p):return json.loads(p.read_text())
def stats(es):
 n=len(es);total=sum(e['generated_tokens'] for e in es)
 return dict(episodes=n,total_generated=total,mean_generated=total/n if n else None)
def main():
 run=ROOT/'results'/NAME;p=read(run/'plan.json');status=read(run/'status.json');assert status['phase']=='complete'
 for path,h in p['hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
 assert not set(p['qids'])&set(p['excluded']) and len(p['qids'])==12
 gold={}
 for split in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
   q=json.loads(line);gold[q['id']]=[q['answer']]+q['answer_aliases']
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1,compute_exact
 pairs=[];gains=set();saving=0;identical=0;es=[];refs=[]
 for q in p['qids']:
  for a in [0,1]:
   e=read(run/'episodes'/f'{q}_a{a}_read.json');ref=read(Path(p['references'][q][str(a)]));assert [v['action'] for v in e['decisions'][:2]]==[a,2]
   for x in [e,ref]:
    assert x['f1']==max(compute_f1(g,x['answer']) for g in gold[q]);x['em']=max(compute_exact(g,x['answer']) for g in gold[q]);assert x['generated_tokens']==sum(v.get('generated_tokens',0) for v in x['reader_calls'])
   epref=[v['messages_sha256'] for v in e['reader_calls'] if v.get('pack_slots')];rpref=[v['messages_sha256'] for v in ref['reader_calls'] if v.get('pack_slots')];assert epref[:2]==rpref[:2]
   assert all(v['scripted_intervention']==(i<2) for i,v in enumerate(e['decisions']))
   for v in e['decisions'][2:]:assert v['action']==max(range(4),key=v['h3_reference_probs'].__getitem__)
   if ref['decisions'][1]['action']==2:
    assert e['answer']==ref['answer'] and e['generated_tokens']==ref['generated_tokens'];identical+=1
   good=not e['errors'] and not ref['errors']
   if good and e['f1']>ref['f1'] and e['em']>=ref['em']:gains.add(q)
   if good and e['f1']==ref['f1'] and e['f1']>0 and e['em']>=ref['em'] and e['generated_tokens']<ref['generated_tokens']:saving+=1
   pairs.append(dict(qid=q,first=a,reference_f1=ref['f1'],two_action_f1=e['f1'],reference_em=ref['em'],two_action_em=e['em'],generated_delta=e['generated_tokens']-ref['generated_tokens']))
   es.append(e);refs.append(ref)
 assert len(es)==24==status['episodes'];assert sum(e['generated_tokens'] for e in es)==status['generated']
 costs={}
 for label,rows in [('same_first_H3_suffix',refs),('search_read_H3_suffix',es)]:costs[label]=dict(f1=sum(e['f1'] for e in rows)/24,all=stats(rows),correct=stats([e for e in rows if e['em']]),incorrect=stats([e for e in rows if not e['em']]),partial=stats([e for e in rows if not e['em'] and 0<e['f1']<1]))
 paired={}
 for a,b,label in [(1,1,'both_correct'),(0,1,'two_action_only_correct'),(1,0,'reference_only_correct'),(0,0,'both_incorrect')]:
  ids=[i for i,(x,y) in enumerate(zip(refs,es)) if x['em']==a and y['em']==b];x=stats([refs[i] for i in ids]);y=stats([es[i] for i in ids]);paired[label]=dict(reference=x,two_action=y,saved_tokens=x['total_generated']-y['total_generated'])
 full=[]
 for q in p['qids']:
  e=read(ROOT/'results/decision_shared_residual72_train_eval_20261024/episodes'/f'h3_{q}.json')
  assert e['f1']==max(compute_f1(g,e['answer']) for g in gold[q]);e['em']=max(compute_exact(g,e['answer']) for g in gold[q]);full.append(e)
 full_reference=dict(f1=sum(e['f1'] for e in full)/12,all=stats(full),correct=stats([e for e in full if e['em']]),incorrect=stats([e for e in full if not e['em']]),scope='One H3 trajectory per question, distinct denominator from24scripted paths')
 report=dict(full_h3_reference=full_reference,verified=True,scope='24 scripted path pairs on12trainingquestions; not learned policy performance or best-of-two accuracy',costs=costs,paired=paired,quality_gain_questions=len(gains),positive_quality_preserving_saving_pairs=saving,reference_read_replay_pairs=identical,diagnostic_gate=len(gains)>=3 and saving>=1,errors=sum(bool(e['errors']) for e in es),pairs=pairs)
 dest=REPO/'lab/reports'/NAME;dest.mkdir(parents=True,exist_ok=True);(dest/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='pairs'},indent=2))
if __name__=='__main__':main()
