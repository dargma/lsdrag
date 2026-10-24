"""One fixed outcome-supervised shared representation/actor fit."""
import json,time
from pathlib import Path
import torch
from torch.nn import functional as F
import numpy as np
from lsdrag.decision_constrained_stop import fit_constrained_stop
from lsdrag.decision_stop_value import stop_margin
from lsdrag.decision_progress_targets import progress_stop_targets as safe_stop_targets
from lsdrag.decision_progress_stop import ProgressStopPolicy,progress_features
import copy,sys
from lsdrag.decision_current_policy_targets import current_policy_targets
from probe_decision_future_critic import ROOT,REPO,sha,save,weights
from evaluate_decision_stop_value_h2 import load_head
from evaluate_decision_progress_stop import load_head as reload_head
NAME='decision_progress_stop120_20261024'
def main():
    p=json.loads((REPO/'lab/registry'/NAME/'plan.json').read_text())
    for path,h in p['hashes'].items():assert sha(Path(path))==h
    torch.set_num_threads(2);torch.manual_seed(20261083);start=time.monotonic()
    fresh=[]
    for path in p['inputs']:fresh.extend(torch.load(path,map_location='cpu',weights_only=True))
    fresh=[r for r in fresh if r['qid'] in p['fit']]
    sys.path.insert(0,str(ROOT.parent/'source/musique_scorer'))
    from metrics.answer import compute_exact,compute_f1
    gold={}
    for split in ['train','dev']:
        for line in (ROOT.parent/f'source/musique/musique_ans_v1.0_{split}.jsonl').open():
            q=json.loads(line);gold[q['id']]=[q['answer']]+q['answer_aliases']
    for r in fresh:
        for sample in r['samples']:
            filename=f"{r['qid']}_t{r['turn']}_a{sample['action']}_r{sample['rep']}.json"
            paths=[ROOT/run/'episodes'/filename for run in p['collections']];paths=[path for path in paths if path.exists()];assert len(paths)==1
            e=json.loads(paths[0].read_text());assert e['f1']==sample['f1'] and e['generated_tokens']==sample['generated']
            assert e['f1']==max(compute_f1(a,e['answer']) for a in gold[r['qid']])
            sample['em']=max(compute_exact(a,e['answer']) for a in gold[r['qid']])
    labels,evidence=safe_stop_targets(fresh);assert evidence['changed_questions']>=3
    retain=[r for i,r in enumerate(fresh) if i not in labels];correction=[fresh[i] for i in labels]
    assert not {r['qid'] for r in retain+correction}&set(p['excluded'])
    out=ROOT/NAME;out.mkdir(exist_ok=False);save(out/'plan.json',p)
    base=load_head(p['reference_spec']);head=ProgressStopPolicy(base)
    frozen={k:v.clone() for k,v in head.anchor.state_dict().items()}
    def tensors(rows):return torch.stack([r['pack'] for r in rows]).float(),torch.stack([r['legal'] for r in rows]),weights(rows)
    allrows=correction+retain
    x,m,w=tensors(allrows)
    with torch.no_grad():
        ref=base(x,m)[0];margin=stop_margin(base,x,m)
        states=head.state(x)
        index={(r['qid'],r['turn']):i for i,r in enumerate(allrows)}
        assert len(index)==len(allrows)
        fs=[]
        for i,r in enumerate(allrows):
            if r['turn']==0:fs.append(progress_features(states[i:i+1],None,None,0))
            else:
                j=index[r['qid'],r['turn']-1]
                previous_action=max(range(4),key=lambda a:allrows[j]['reference_probs'][a])
                fs.append(progress_features(states[i:i+1],states[j:j+1],previous_action,r['turn']))
        features=torch.cat(fs)
        target=margin.sigmoid()
        target[:len(correction)]=.5*target[:len(correction)]+.5*torch.tensor([int(a==3) for a in labels.values()])
        odds=ref[:,3]-torch.logsumexp(ref[:,:3],dim=1)
    lossw=torch.cat([.5*weights(correction),.5*weights(retain)])
    delta,solver=fit_constrained_stop(features.numpy(),margin.numpy(),target.numpy(),lossw.numpy(),range(len(correction),len(allrows)),odds.numpy(),w.numpy())
    with torch.no_grad():
        head.progress.weight.copy_(torch.tensor(delta,dtype=head.progress.weight.dtype).reshape(1,-1))
        output=ref.clone();output[:,3]+=head.progress(features).squeeze(-1)
        initials=torch.tensor([r['turn']==0 for r in allrows])
        assert torch.equal(output[initials],ref[initials])
        unchanged=torch.equal(output[len(correction):].argmax(-1),ref[len(correction):].argmax(-1))
        assert torch.equal(output[:,:3],ref[:,:3])
        kl=float((w*(ref.softmax(-1)*(ref.log_softmax(-1)-output.log_softmax(-1))).sum(-1)).sum())
        matches=output[:len(correction)].argmax(-1)==torch.tensor(list(labels.values()))
        changedq={correction[i]['qid'] for i in range(len(correction)) if matches[i]}
    assert all(torch.equal(v,frozen[k]) for k,v in head.anchor.state_dict().items())
    progress=solver['success'] and solver['constraint_violation']<=1e-7 and unchanged and kl<=.0100001 and len(changedq)>=3
    torch.save(head.state_dict(),out/'action.pt');spec=dict(kind='progress_stop',checkpoint=str(out/'action.pt'),sha256=sha(out/'action.pt'))
    restored=reload_head(spec);assert all(torch.equal(v,restored.state_dict()[k]) for k,v in head.state_dict().items())
    result=dict(candidate=spec,learning_progress=progress,supported_matches=int(matches.sum()),changed_questions=len(changedq),solver=solver,fp32_kl=kl,protected_actions_unchanged=unchanged,corrections=len(correction),retention=len(retain),target_evidence=evidence,anchor_unchanged=True,trainable_parameters=132,seconds=time.monotonic()-start)
    save(out/'summary.json',result);save(out/'status.json',dict(phase='complete',pid=None));save(REPO/'lab/reports'/NAME/'summary.json',result)
    torch.save(dict(features=features,reference_logits=ref,legal=m),out/'fit_features.pt')
    save(out/'state_check.json',[dict(qid=r['qid'],turn=r['turn'],prefix_sha256=r['prefix_sha256'],protected=i>=len(correction),before=int(ref[i].argmax()),after=int(output[i].argmax()),target=labels[list(labels)[i]] if i<len(correction) else None) for i,r in enumerate(allrows)])
    print(json.dumps({k:v for k,v in result.items() if k!='target_evidence'},indent=2))
if __name__=='__main__':main()
