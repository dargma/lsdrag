"""Frozen training-only residual permutation diagnostic, no new reader calls."""
import hashlib,json,sys
from pathlib import Path
import torch
from evaluate_decision_stop_value_h2 import load_head
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[2]
NAME='decision_h3_state_correction72_20261024'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n')
def main():
 plan=read(REPO/'lab/registry'/NAME/'plan.json')
 for p,h in plan['hashes'].items():assert sha(Path(p))==h,p
 out=ROOT/'results'/NAME;out.mkdir(exist_ok=False)
 torch.set_num_threads(2)
 records=[]
 for p in plan['inputs']:records.extend(torch.load(p,weights_only=True,map_location='cpu'))
 assert len({(r['qid'],r['prefix_sha256']) for r in records})==len(records)
 rows=[]
 for qid in sorted(plan['fit'],key=lambda q:hashlib.sha256(('20261082:'+q).encode()).hexdigest()):
  valid=[r for r in records if r['qid']==qid and not any(s['errors'] for s in r['samples'])]
  assert valid,qid
  rows.append(min(valid,key=lambda r:hashlib.sha256(('20261082:'+r['prefix_sha256']).encode()).hexdigest()))
 assert len(rows)==72
 head=load_head(plan['head']);x=torch.stack([r['pack'] for r in rows]);legal=torch.stack([r['legal'] for r in rows])
 with torch.no_grad():
  base=head.base(x,legal)[0];actual=head(x,legal)[0];residual=actual[:,3]-base[:,3]
 assert torch.equal(base[:,:3],actual[:,:3])
 expected=torch.tensor([r['reference_probs'] for r in rows])
 assert torch.allclose(actual.softmax(-1),expected,atol=1e-5,rtol=1e-5)
 sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_exact,compute_f1
 gold={}
 for split in ['train','dev']:
  for line in (ROOT/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
   q=json.loads(line);gold[q['id']]=[q['answer']]+q['answer_aliases']
 q=torch.full((72,4),float('nan'));cost=q.clone();em=q.clone();hashes={}
 for i,r in enumerate(rows):
  assert {s['action'] for s in r['samples']}==set(torch.where(legal[i])[0].tolist())
  for s in r['samples']:
   filename=f"{r['qid']}_t{r['turn']}_a{s['action']}_r{s['rep']}.json"
   paths=[ROOT/'results'/run/'episodes'/filename for run in plan['collections']];paths=[p for p in paths if p.exists()];assert len(paths)==1
   p=paths[0];e=read(p);assert e['f1']==s['f1'] and e['generated_tokens']==s['generated'] and not e['errors']
   assert e['f1']==max(compute_f1(a,e['answer']) for a in gold[r['qid']])
   q[i,s['action']]=e['f1'];cost[i,s['action']]=e['generated_tokens'];em[i,s['action']]=max(compute_exact(a,e['answer']) for a in gold[r['qid']]);hashes[str(p)]=sha(p)
 idx=torch.arange(72);a=actual.argmax(-1)
 def stats(actions):
  f=q[idx,actions];c=cost[idx,actions];correct=em[idx,actions].bool()
  return dict(f1=float(f.mean()),mean_generated=float(c.mean()),correct=int(correct.sum()),correct_total_generated=float(c[correct].sum()),incorrect_total_generated=float(c[~correct].sum()),partial_questions=int(((f>0)&(f<1)).sum()),actions=actions.tolist())
 controls=[];changed=set()
 for shift in range(1,72):
  logits=base.clone();logits[:,3]+=residual.roll(shift);actions=logits.argmax(-1)
  changed.update(torch.where(actions!=a)[0].tolist());s=stats(actions);s['shift']=shift;s['changed_questions']=int((actions!=a).sum())
  # Same-question EM accounting: actual H3 is candidate, shuffled policy reference.
  ec=em[idx,actions].bool();ac=em[idx,a].bool();delta=cost[idx,actions]-cost[idx,a]
  s['paired_saved_tokens']={name:float(delta[mask].sum()) for name,mask in [('both_correct',ec&ac),('h3_only_correct',~ec&ac),('control_only_correct',ec&~ac),('both_incorrect',~ec&~ac)]}
  controls.append(s)
 observed=stats(a);mean={k:sum(s[k] for s in controls)/71 for k in ['f1','mean_generated','correct','correct_total_generated','incorrect_total_generated','changed_questions']}
 coverage=len(changed)>=10
 dominates=observed['f1']>=mean['f1'] and observed['mean_generated']<=mean['mean_generated'] and (observed['f1']>mean['f1'] or observed['mean_generated']<mean['mean_generated'])
 summary=dict(scope='72 training questions, one hash-selected state each, one-action intervention then measured H3 suffix; not full-policy or independent evaluation. All71 nonidentity cyclic shifts of residuals; descriptive, not a permutation p-value.',actual=observed,shuffled_mean=mean,changed_questions_union=len(changed),coverage_sufficient=coverage,diagnostic_gate=coverage and dominates,mean_paired_saved_tokens={k:sum(s['paired_saved_tokens'][k] for s in controls)/71 for k in controls[0]['paired_saved_tokens']},new_reader_calls=0)
 save(out/'summary.json',summary);save(out/'controls.json',controls);save(out/'provenance.json',dict(plan=plan,rows=[dict(qid=r['qid'],turn=r['turn'],prefix_sha256=r['prefix_sha256'],residual=float(residual[i])) for i,r in enumerate(rows)],episode_sha256=hashes));save(out/'status.json',dict(phase='complete',pid=None))
 for name in ['summary','controls','provenance']:save(REPO/'lab/reports'/NAME/(name+'.json'),read(out/(name+'.json')))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
