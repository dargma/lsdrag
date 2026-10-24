"""One frozen outcome-preference fit and its state-independent control."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter
import torch
from torch.nn import functional as F
from evaluate_decision_stop_value_h2 import load_head
from lsdrag.decision_tool_return import ToolReturnPolicy
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[1]
NAME='decision_tool_return_final_20261024'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=json.loads((REPO/'lab/registry'/NAME/'plan.json').read_text())
    for f,h in p['hashes'].items():assert sha(f)==h,f
    torch.set_num_threads(2);torch.manual_seed(20261088)
    sys.path.insert(0,str(ROOT/'source/musique_scorer'));from metrics.answer import compute_exact,compute_f1
    gold={q['id']:[q['answer']]+q['answer_aliases'] for q in map(json.loads,(ROOT/'source/musique/musique_ans_v1.0_train.jsonl').open())}
    rows=[];moves=[]
    for source in p['inputs']:
        for r in torch.load(source,weights_only=True,map_location='cpu'):
            if r['qid'] not in p['fit']:continue
            anchor=max(range(4),key=lambda a:r['reference_probs'][a])
            if anchor==3:continue
            samples={s['action']:s for s in r['samples']};assert len(samples)==len(r['samples'])
            assert set(samples)==set(torch.where(r['legal'])[0].tolist())
            for a,s in samples.items():
                paths=[ROOT/'results'/c/'episodes'/f"{r['qid']}_t{r['turn']}_a{a}_r{s['rep']}.json" for c in p['collections']];paths=[x for x in paths if x.exists()];assert len(paths)==1
                e=json.loads(paths[0].read_text());assert e['generated_tokens']==s['generated'] and e['f1']==s['f1']
                assert abs(e['f1']-max(compute_f1(x,e['answer']) for x in gold[r['qid']]))<1e-10
                s['em']=max(compute_exact(x,e['answer']) for x in gold[r['qid']])
            ref=samples[anchor];target=anchor
            choices=[a for a,s in samples.items() if a!=3 and not s['errors'] and not ref['errors'] and s['em']>=ref['em'] and s['f1']>=ref['f1'] and s['f1']>0 and s['generated']<=ref['generated']]
            if choices:target=min(choices,key=lambda a:(-samples[a]['f1'],samples[a]['generated'],a!=anchor,a))
            if target!=anchor:moves.append(dict(qid=r['qid'],turn=r['turn'],before=anchor,after=target,f1_delta=samples[target]['f1']-ref['f1'],generated_delta=samples[target]['generated']-ref['generated']))
            rows.append((r,anchor,target))
    assert len({m['qid'] for m in moves})>=3
    packs=torch.stack([r['pack'] for r,_,_ in rows]).float();legal=torch.stack([r['legal'] for r,_,_ in rows]);labels=torch.tensor([t for _,_,t in rows]);changed=torch.tensor([a!=t for _,a,t in rows])
    # Equal question mass within changed and retained categories, each half.
    weights=torch.zeros(len(rows))
    for flag in [True,False]:
        idx=[i for i in range(len(rows)) if bool(changed[i])==flag];counts=Counter(rows[i][0]['qid'] for i in idx)
        for i in idx:weights[i]=.5/len(counts)/counts[rows[i][0]['qid']]
    base=load_head(p['reference_spec']);out=ROOT/'results'/NAME;out.mkdir(exist_ok=False)
    evidence=dict(states=len(rows),changed_states=int(changed.sum()),changed_questions=len({m['qid'] for m in moves}),moves=moves,models={})
    for name,constant in [('state',False),('constant',True)]:
        model=ToolReturnPolicy(base,constant=constant)
        with torch.no_grad():
            old=base(packs,legal)[0];prob=old[:,:3].softmax(-1)
            assert torch.equal(model(packs,legal)[0].argmax(-1),old.argmax(-1))
        opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=.003,weight_decay=.01)
        for epoch in range(200):
            logits,_=model.tool_logits(packs,legal);logp=logits.log_softmax(-1)
            ce=F.cross_entropy(logits,labels,reduction='none');kl=(prob*(prob.clamp_min(1e-9).log()-logp)).sum(-1)
            loss=(weights*(ce+.1*kl)).sum();opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            pred=model(packs,legal)[0];assert torch.equal(pred.argmax(-1)==3,old.argmax(-1)==3)
            # Test stop invariance on every stored state, including FINAL states.
            allrows=[r for source in p['inputs'] for r in torch.load(source,weights_only=True,map_location='cpu') if r['qid'] in p['fit']]
            x=torch.stack([r['pack'] for r in allrows]).float();mask=torch.stack([r['legal'] for r in allrows]);assert torch.equal(model(x,mask)[0].argmax(-1)==3,base(x,mask)[0].argmax(-1)==3)
            for k,v in base.state_dict().items():assert torch.equal(v,model.anchor.state_dict()[k])
        dest=out/f'{name}.pt';torch.save(model.state_dict(),dest)
        evidence['models'][name]=dict(sha256=sha(dest),train_target_matches=int((pred.argmax(-1)==labels).sum()),changed_target_matches=int(((pred.argmax(-1)==labels)&changed).sum()),loss=float(loss),parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),stop_boundary_verified_states=len(allrows))
    (out/'fit.json').write_text(json.dumps(evidence,indent=2)+'\n');(out/'status.json').write_text(json.dumps(dict(phase='complete',pid=None))+'\n')
    dest=REPO/'lab/reports'/NAME;dest.mkdir(parents=True,exist_ok=True);(dest/'fit.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
