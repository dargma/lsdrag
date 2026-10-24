"""Three-way same9B evaluation and qualified historical4B trend comparison."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(es):
 n=len(es);total=sum(e['generated_tokens'] for e in es)
 return dict(n=n,total_generated=total,mean_generated=total/n if n else None)
def main():
 run=ROOT/'results/qwen35_9b_h3_validation64_20261024';spec=read(run/'spec.json');status=read(run/'status.json');assert status['phase']=='complete' and status['episodes']==192
 plan=read(REPO/'lab/registry/qwen35_9b_h3_20261024/training_plan.json');assert spec['qids']==plan['development64']
 protected=read(REPO/'lab/registry/decision_shared_final_20261023/selection_plan.json');assert not set(spec['qids'])&set(protected['reserved_qids']+protected['test_qids']+plan['train'])
 for f,h in spec['source_sha256'].items():assert sha(REPO/f)==h
 for v in spec['policies'].values():assert sha(v['checkpoint'])==v['sha256']
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1,compute_exact
 gold={q['id']:[q['answer']]+q['answer_aliases'] for q in map(json.loads,(ROOT/'source/musique/musique_ans_v1.0_dev.jsonl').open())};rows={};hashes={};errors=0;total=0
 for path in sorted((run/'episodes').glob('*.json')):
  e=read(path);assert e['qid'] in spec['qids'] and e['policy'] in spec['policies'];assert abs(e['f1']-max(compute_f1(a,e['answer']) for a in gold[e['qid']]))<1e-10
  assert e['generated_tokens']==sum(u.get('generated_tokens',0) for u in e['reader_calls'])
  for u in e['reader_calls']:
   if 'token_ids' in u:assert len(u['token_ids'])==u['generated_tokens']
  e['em']=max(compute_exact(a,e['answer']) for a in gold[e['qid']]);key=(e['policy'],e['qid']);assert key not in rows;rows[key]=e;hashes[str(path)]=sha(path);errors+=bool(e['errors']);total+=e['generated_tokens']
 assert len(rows)==192 and total==status['generated'];policies={}
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(spec['model'],local_files_only=True)
 for name in spec['policies']:
  es=[rows[name,q] for q in spec['qids']];correct=[e for e in es if e['em']];incorrect=[e for e in es if not e['em']];actions={str(a):sum(d['action']==a for e in es for d in e['decisions']) for a in range(4)}
  texts=[tokenizer.decode(u['token_ids'],skip_special_tokens=False) for e in es for u in e['reader_calls'] if 'token_ids' in u]
  policies[name]=dict(f1=sum(e['f1'] for e in es)/64,all=stats(es),correct=stats(correct),incorrect=stats(incorrect),partial=stats([e for e in incorrect if 0<e['f1']<1]),f1_one_nonexact=stats([e for e in incorrect if e['f1']==1]),mean_tools=sum(e['tool_calls'] for e in es)/64,forced_final=sum(e['forced_final'] for e in es),actions=actions,output_utf8_bytes=sum(len(x.encode()) for x in texts),output_characters=sum(len(x) for x in texts))
 paired={}
 for ref,candidate in [('base','h3'),('constant','h3'),('base','constant')]:
  groups={}
  for a,b,label in [(1,1,'both_correct'),(0,1,'candidate_only_correct'),(1,0,'reference_only_correct'),(0,0,'both_incorrect')]:
   qs=[q for q in spec['qids'] if rows[ref,q]['em']==a and rows[candidate,q]['em']==b];old=stats([rows[ref,q] for q in qs]);new=stats([rows[candidate,q] for q in qs]);groups[label]=dict(reference=old,candidate=new,saved_tokens=old['total_generated']-new['total_generated'])
  paired[f'{candidate}_vs_{ref}']=dict(groups=groups,f1_delta=policies[candidate]['f1']-policies[ref]['f1'],token_ratio=policies[candidate]['all']['total_generated']/policies[ref]['all']['total_generated'],f1_wins=sum(rows[candidate,q]['f1']>rows[ref,q]['f1'] for q in spec['qids']),f1_losses=sum(rows[candidate,q]['f1']<rows[ref,q]['f1'] for q in spec['qids']))
 stop_divergences={}
 for candidate in ['h3','constant']:
  changed=[]
  for q in spec['qids']:
   a,b=rows['base',q],rows[candidate,q]
   for i,(x,y) in enumerate(zip(a['decisions'],b['decisions'])):
    if x['action']!=y['action']:
     assert 3 in [x['action'],y['action']], 'First same-state difference must be stopping'
     assert all(all(u[k]==v[k] for k in ['tool_name','arguments','result']) for u,v in zip(a['trajectory'][:i],b['trajectory'][:i]))
     changed.append(dict(qid=q,turn=i,base=x['action'],candidate=y['action']));break
  stop_divergences[candidate]=changed
 report=dict(stop_divergences=stop_divergences,verified=True,scope='Repeated development64; no independent confirmation or causal scale claim',errors=errors,policies=policies,paired=paired,episode_sha256=hashes,interpretation='Compare H3 against same9B outcome base and constant-bias control. Historical4B base had different exploratory preparation; tokenizers/architecture/generation/size also change. Completion is valid experiment/report, not forced positive improvement.')
 out=REPO/'lab/reports/qwen35_9b_h3_validation64_20261024';out.mkdir(parents=True,exist_ok=True);(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='episode_sha256'},indent=2))
if __name__=='__main__':main()
