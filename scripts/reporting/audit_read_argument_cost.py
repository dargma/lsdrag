"""Read-only cost feasibility: actual raw decode tokens, no savings claim."""
import json,sys,hashlib,math
from pathlib import Path
from collections import Counter
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[2]
def main():
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_exact,compute_f1
 gold={}
 for split in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
   q=json.loads(line);gold[q['id']]=[q['answer']]+q['answer_aliases']
 reports={};hashes={}
 for name,run,prefix in [('h2','decision_stop_value_h2_validation64_20261024','stop_value'),('h3','decision_budget_stop_h3_validation64_20261024','budget_stop')]:
  sums=Counter();counts=Counter();groups={k:dict(questions=0,total_generated=0,tokens_by_action=Counter()) for k in ['correct','incorrect','partial']};f1=0.;n=0
  for path in (ROOT/'results'/run/'episodes').glob(prefix+'_*.json'):
   raw=path.read_bytes();hashes[str(path)]=hashlib.sha256(raw).hexdigest();e=json.loads(raw);assert e['f1']==max(compute_f1(a,e['answer']) for a in gold[e['qid']]);em=max(compute_exact(a,e['answer']) for a in gold[e['qid']]);f1+=e['f1'];n+=1
   calls=[c for c in e['reader_calls'] if 'raw_json' in c];assert len(calls)==len(e['decisions']);assert sum(c.get('generated_tokens',0) for c in calls)==e['generated_tokens']
   labels=['correct' if em else 'incorrect']+(['partial'] if not em and 0<e['f1']<1 else [])
   for label in labels:groups[label]['questions']+=1;groups[label]['total_generated']+=e['generated_tokens']
   for d,c in zip(e['decisions'],calls):
    action=d['action'];sums[action]+=c['generated_tokens'];counts[action]+=1
    for label in labels:groups[label]['tokens_by_action'][action]+=c['generated_tokens']
  assert n==64
  for group in groups.values():group['mean_generated']=group['total_generated']/group['questions'] if group['questions'] else None
  reports[name]=dict(f1=f1/n,questions=n,total_generated=sum(sums.values()),mean_generated=sum(sums.values())/n,tokens_by_action=dict(sums),calls_by_action=dict(counts),correctness=groups,all_read_generation_free_upper_bound_mean=(sum(sums.values())-sums[2])/n)
 total=reports['h2']['total_generated'];target=.7*6298;needed=math.ceil(total-target)
 result=dict(scope='Repeated development cost profile; zero-read-generation number is an unattained upper bound, not measured acceleration. No output token relabeling permitted.',policies=reports,h2_tokens_to_remove_for_fixed_goal=needed,h2_fraction_read_argument_generation_to_remove=needed/reports['h2']['tokens_by_action'][2],episode_sha256=hashes)
 dest=REPO/'lab/reports/decision_argument_cost_feasibility_20261024';dest.mkdir(parents=True,exist_ok=True);(dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='episode_sha256'},indent=2))
if __name__=='__main__':main()
