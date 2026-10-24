"""Rescore historical4B base/H3 for matched-question trend, not scale causality."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 run=ROOT/'results/decision_budget_stop_h3_validation64_20261024';spec=read(run/'spec.json');plan=read(REPO/'lab/registry/qwen35_9b_h3_20261024/training_plan.json');assert spec['qids']==plan['development64']
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_f1,compute_exact
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(spec['model'],local_files_only=True)
 gold={q['id']:[q['answer']]+q['answer_aliases'] for source_split in ['train','dev'] for q in map(json.loads,(ROOT/f'source/musique/musique_ans_v1.0_{source_split}.jsonl').open())}
 rows={};hashes={};policies={}
 def stats(es):
  n=len(es);g=sum(e['generated_tokens'] for e in es);return dict(n=n,total_generated=g,mean_generated=g/n if n else None)
 for original,name in [('selected_reference','base'),('budget_stop','h3')]:
  es=[]
  for qid in spec['qids']:
   path=run/'episodes'/f'{original}_{qid}.json';e=read(path);assert e['f1']==max(compute_f1(a,e['answer']) for a in gold[qid]);e['em']=max(compute_exact(a,e['answer']) for a in gold[qid]);assert e['generated_tokens']==sum(u.get('generated_tokens',0) for u in e['reader_calls']);hashes[str(path)]=sha(path);rows[name,qid]=e;es.append(e)
  correct=[e for e in es if e['em']];incorrect=[e for e in es if not e['em']];texts=[tokenizer.decode(u['token_ids'],skip_special_tokens=False) for e in es for u in e['reader_calls'] if 'token_ids' in u]
  policies[name]=dict(f1=sum(e['f1'] for e in es)/64,all=stats(es),correct=stats(correct),incorrect=stats(incorrect),partial=stats([e for e in incorrect if 0<e['f1']<1]),actions=dict(Counter(d['action'] for e in es for d in e['decisions'])),mean_tools=sum(e['tool_calls'] for e in es)/64,forced_final=sum(e['forced_final'] for e in es),output_characters=sum(len(t) for t in texts),output_utf8_bytes=sum(len(t.encode()) for t in texts),errors=sum(bool(e['errors']) for e in es))
 groups={}
 for a,b,label in [(1,1,'both_correct'),(0,1,'h3_only_correct'),(1,0,'base_only_correct'),(0,0,'both_incorrect')]:
  ids=[q for q in spec['qids'] if rows['base',q]['em']==a and rows['h3',q]['em']==b];old=stats([rows['base',q] for q in ids]);new=stats([rows['h3',q] for q in ids]);groups[label]=dict(base=old,h3=new,saved_tokens=old['total_generated']-new['total_generated'])
 out=REPO/'lab/reports/qwen35_9b_historical_reference_20261024';out.mkdir(parents=True,exist_ok=True);report=dict(verified=True,scope='Historical4B same64questions; base preparation, architecture, generation and tokenizer differ from new9B. Within-model effects primary; no causal scale attribution.',model=spec['model'],qids=spec['qids'],policies=policies,paired=groups,f1_delta=policies['h3']['f1']-policies['base']['f1'],token_ratio=policies['h3']['all']['total_generated']/policies['base']['all']['total_generated'],episode_sha256=hashes);(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['episode_sha256','qids']},indent=2))
if __name__=='__main__':main()
