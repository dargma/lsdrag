"""Independent metric calculation plus checkpoint/prediction/split/backup audit."""
import hashlib,json,math,tarfile
from collections import defaultdict,Counter
from pathlib import Path
import torch
from lsdrag.decision_future_critic import FutureCritic
ROOT=Path('/content/local_fast/lsdrag-text-v1/results')
REPO=Path('/content/local_fast/lsdrag-code')
NAME='decision_future_critic_probe72_20261024'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(2)
    run=ROOT/NAME;p=read(run/'plan.json');s=read(run/'summary.json');pred=read(run/'predictions.json')
    assert read(run/'status.json')['phase']=='complete'
    assert not p['pending_completed_collection_hash']
    for f,h in p['input_sha256'].items():assert sha(Path(f))==h
    for f,h in p['source_sha256'].items():assert sha(REPO/f)==h
    collection=read(ROOT/'decision_h3_diversity48_20261024/manifest.json')
    assert p['fit']==collection['critic_fit'] and p['monitor']==collection['critic_monitor']
    assert not set(p['fit'])&set(p['monitor']+p['excluded_prior_monitor'])
    assert not set(p['fit']+p['monitor'])&set(collection['validation']+collection['test'])
    raw=[]
    for f in p['inputs']:raw.extend(torch.load(f,map_location='cpu',weights_only=True))
    usable=[r for r in raw if not any(a['errors'] for a in r['samples'])]
    fit=[r for r in usable if r['qid'] in p['fit']];monitor={(r['qid'],r['turn']):r for r in usable if r['qid'] in p['monitor']}
    assert len(pred)==len(monitor) and {(r['qid'],r['turn']) for r in pred}==set(monitor)
    counts=Counter(r['qid'] for r in fit);means=defaultdict(lambda:[0.,0.,0.])
    for r in fit:
        w=1/len(counts)/counts[r['qid']]/len(r['samples'])
        for a in r['samples']:
            v=means[a['action']];v[0]+=w*a['f1'];v[1]+=w*a['generated'];v[2]+=w
    packs=torch.stack([monitor[r['qid'],r['turn']]['pack'] for r in pred])
    for checkpoint,prefix in [(run/'critic.pt','predicted'),(Path(p['old_checkpoint']),'old')]:
        model=FutureCritic();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
        with torch.no_grad():q,c,a=model(packs)
        assert torch.count_nonzero(a)==0
        for i,row in enumerate(pred):
            source=monitor[row['qid'],row['turn']];assert row['prefix_sha256']==source['prefix_sha256']
            by={a['action']:a for a in source['samples']};assert set(by)=={b['action'] for b in row['branches']}
            for b in row['branches']:
                j=b['action'];assert b['f1']==by[j]['f1'] and b['generated']==by[j]['generated']
                assert math.isclose(float(q[i,j]),b[prefix+'_f1'],abs_tol=1e-6)
                assert math.isclose(float(c[i,j]),b[prefix+'_generated'],abs_tol=1e-4)
                assert math.isclose(b['baseline_f1'],means[j][0]/means[j][2],abs_tol=1e-6)
                assert math.isclose(b['baseline_generated'],means[j][1]/means[j][2],abs_tol=1e-4)
    avg=lambda a:sum(a)/len(a);computed={}
    for label,prefix in [('expanded','predicted'),('old','old'),('baseline','baseline')]:
        perq=defaultdict(list);pairs=0;informative=set()
        for row in pred:
            b=row['branches'];gaps=[];sign=[]
            for i,a in enumerate(b):
                for c in b[i+1:]:
                    t=a['f1']-c['f1'];v=a[prefix+'_f1']-c[prefix+'_f1'];gaps.append((t-v)**2)
                    if abs(t)>1e-8:sign.append(float(t*v>0));pairs+=1;informative.add(row['qid'])
            perq[row['qid']].append((avg(gaps),avg([abs(a[prefix+'_generated']-a['generated']) for a in b]),avg(sign) if sign else None))
        result=dict(gap_mse=avg([avg([a[0] for a in r]) for r in perq.values()]),cost_mae=avg([avg([a[1] for a in r]) for r in perq.values()]),gap_sign_accuracy=avg([avg([a[2] for a in r if a[2] is not None]) for r in perq.values() if any(a[2] is not None for a in r)]),informative_pairs=pairs,informative_questions=len(informative))
        for k,v in result.items():assert math.isclose(v,s['results'][label][k],abs_tol=1e-5)
        computed[label]=result
    a=computed['expanded'];enough=a['informative_pairs']>=10 and a['informative_questions']>=5
    gate=enough and a['gap_sign_accuracy']>.5 and all(a[k]<computed[b][k] for b in ['old','baseline'] for k in ['gap_mse','cost_mae'])
    assert gate==s['gate_passed'] and enough==s['coverage_sufficient']
    assert sha(run/'critic.pt')==s['checkpoint_sha256']
    archive=read(REPO/'lab/registry'/NAME/'archive.json');assert sha(Path(archive['path']))==archive['sha256']
    with tarfile.open(archive['path']) as tf:
        for member in tf.getmembers():
            if member.isfile():assert hashlib.sha256(tf.extractfile(member).read()).hexdigest()==sha(run.joinpath(*Path(member.name).parts[1:]))
    report=dict(verified=True,metrics=computed,gate_passed=gate,actor_zero_verified=True,checkpoint_replay=True,split_verified=True,archive_sha256=archive['sha256'],scope='Critic predictive transfer only; no learned-policy improvement claim')
    (REPO/'lab/reports'/NAME/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
