"""Frozen first81 training episodes: descriptive output audit, no tuning labels."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter,defaultdict
from lsdrag.text_decision_benchmark import canonical
from lsdrag.text_integrated import normalize_args
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[2]
def main():
 run=ROOT/'results/qwen35_9b_teacher192_20261024';spec=json.loads((run/'spec.json').read_text());qids=spec['qids'][:81]
 raw={q['id']:q for q in map(json.loads,(ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open())}
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_exact,compute_f1
 groups=defaultdict(list);cases=[];hashes={};counts=Counter();allrows=[]
 for qid in qids:
  path=run/'episodes'/f'native_teacher_{qid}.json';e=json.loads(path.read_text());q=raw[qid];answers=[q['answer']]+q['answer_aliases'];em=max(compute_exact(a,e['answer']) for a in answers);assert e['f1']==max(compute_f1(a,e['answer']) for a in answers)
  group='correct' if em else 'partial' if e['f1']>0 else 'zero_f1';groups[group].append(e);allrows.append(e);hashes[qid]=hashlib.sha256(path.read_bytes()).hexdigest()
  counts['forced_final']+=e['forced_final'];counts['empty_answer']+=not bool(e['answer'].strip());counts['answers_over10_words']+=len(e['answer'].split())>10;counts['first_keyword']+=e['decisions'][0]['action']==0;counts['episodes_without_read']+=not any(t['tool_name']=='read_chunk' for t in e['trajectory']);counts['errors']+=bool(e['errors'])
  seen_calls=set();observed=set();repeat=False
  for t in e['trajectory']:
   key=(t['tool_name'],canonical(normalize_args(t['tool_name'],t['arguments'])));ids={v['chunk_id'] for v in t['result'] if isinstance(v,dict) and 'chunk_id' in v}
   if key in seen_calls:counts['identical_repeated_calls']+=1;repeat=True
   if t['tool_name'].endswith('search'):
    counts['search_calls']+=1;counts['nonempty_search_no_new_ids']+=bool(ids) and not bool(ids-observed)
   seen_calls.add(key);observed|=ids
  counts['episodes_with_identical_repeat']+=repeat
  if len(groups[group])<=3:
   cases.append(dict(group=group,qid=qid,question=q['question'],gold=q['answer'],answer=e['answer'],f1=e['f1'],forced_final=e['forced_final'],generated=e['generated_tokens'],trajectory=[dict(tool=t['tool_name'],arguments=t['arguments'],titles=[v.get('title') for v in t['result'] if isinstance(v,dict)]) for t in e['trajectory']]))
 def stats(es):
  return dict(n=len(es),generated_total=sum(e['generated_tokens'] for e in es),generated_mean=sum(e['generated_tokens'] for e in es)/len(es),forced_final=sum(e['forced_final'] for e in es),mean_tools=sum(e['tool_calls'] for e in es)/len(es))
 incorrect=groups['partial']+groups['zero_f1'];report=dict(scope='Frozen first81 TRAIN teacher episodes; current common-protocol9B, not learnedH3. First3 chronological examples per exact/partial/zeroF1 stratum; no configuration change or evaluation relabeling.',counts=dict(counts),groups={k:stats(v) for k,v in groups.items()},correctness={'correct':stats(groups['correct']),'incorrect':stats(incorrect),'partial_incorrect':stats(groups['partial'])},mean_answer_words=sum(len(e['answer'].split()) for e in allrows)/81,cases=cases,episode_sha256=hashes)
 out=REPO/'lab/reports/qwen35_9b_qualitative_prefix81_20261024';out.mkdir(parents=True,exist_ok=True);(out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['cases','episode_sha256']},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
