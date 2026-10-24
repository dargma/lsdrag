"""Single fresh9B base-policy update and original H3 stopping fit."""
import argparse,copy,hashlib,json
from pathlib import Path
from collections import defaultdict
import torch
from torch.nn import functional as F
from lsdrag.decision_state_learning import StateHead
from lsdrag.decision_stop_value import StopValueHead,stop_margin
from lsdrag.decision_budget_targets import budget_targets
from lsdrag.decision_policy_followup import state_weights
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[1];REG=REPO/'lab/registry/qwen35_9b_h3_20261024'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def base_load(p):
    h=StateHead(dim=4096,inputs='multi',body='mlp');h.load_state_dict(torch.load(p,weights_only=True,map_location='cpu'));return h.eval()
def teacher_retention():
    run=ROOT/'results/qwen35_9b_teacher192_20261024';records=torch.load(run/'teacher_states.pt',weights_only=True,map_location='cpu');offset=defaultdict(int);prefixes={}
    for r in records:
        q=r['qid']
        if q not in prefixes:
            ep=read(run/'episodes'/f'native_teacher_{q}.json');prefills=[u for u in ep['reader_calls'] if u.get('pack_slots')];assert len(prefills)==len(ep['decisions'])
            prefixes[q]=[u['messages_sha256'] for u,d in zip(prefills,ep['decisions']) if not d['forced']]
        r['prefix_sha256']=prefixes[q][offset[q]];offset[q]+=1
    assert all(offset[q]==len(v) for q,v in prefixes.items())
    return records

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['base','stop'],required=True);args=ap.parse_args()
    plan=read(REG/'training_plan.json');torch.set_num_threads(2)
    run=ROOT/'results'/('qwen35_9b_outcomes48_20261024' if args.stage=='base' else 'qwen35_9b_stop48_20261024')
    assert read(run/'status.json')['phase']=='complete'
    audit=read(REPO/'lab/reports'/run.name/'verification.json');assert audit['verified'] and audit['tensor_sha256']==sha(run/'states.pt')
    records=torch.load(run/'states.pt',weights_only=True,map_location='cpu');spec=read(run/'spec.json');base=base_load(spec['checkpoint']);assert sha(spec['checkpoint'])==spec['hashes'][spec['checkpoint']]
    if args.stage=='base':
        torch.manual_seed(20261066);labels={};moves=[]
        for i,r in enumerate(records):
            samples={s['action']:s for s in r['samples']};anchor=max(range(4),key=lambda a:r['reference_probs'][a]);ref=samples[anchor]
            choices=[a for a,s in samples.items() if not s['errors'] and not ref['errors'] and s['em']>=ref['em'] and s['f1']>=ref['f1'] and s['f1']>0 and (s['f1']>ref['f1'] or s['generated']<ref['generated'])]
            if choices:
                target=min(choices,key=lambda a:(-samples[a]['f1'],samples[a]['generated'],a))
                if target!=anchor:labels[i]=target;moves.append(dict(qid=r['qid'],turn=r['turn'],before=anchor,after=target))
        evidence=dict(changed_questions=len({r['qid'] for r in moves}),moves=moves)
    else:
        torch.manual_seed(20261069);labels,evidence=budget_targets(records)
    assert evidence['changed_questions']>=3,'Insufficient supported changes; do not silently retune'
    seen={(r['qid'],r['prefix_sha256']) for r in records}
    retain=[r for r in teacher_retention() if (r['qid'],r['prefix_sha256']) not in seen]+[r for i,r in enumerate(records) if i not in labels]
    corrections=[records[i] for i in labels]
    def tensors(rows):return torch.stack([r['pack'] for r in rows]).float(),torch.stack([r['legal'] for r in rows]),state_weights([r['qid'] for r in rows],'cpu')
    cx,cm,cw=tensors(corrections);rx,rm,rw=tensors(retain)
    with torch.no_grad():
        if args.stage=='base':
            target=.5*base(cx,cm)[0].softmax(-1)+.5*F.one_hot(torch.tensor(list(labels.values())),4).float();rt=base(rx,rm)[0].softmax(-1)
        else:target=.5*stop_margin(base,cx,cm).sigmoid()+.5*torch.tensor(list(labels.values()));rt=stop_margin(base,rx,rm).sigmoid()
    out=ROOT/'results'/('qwen35_9b_base_20261024' if args.stage=='base' else 'qwen35_9b_h3_20261024');out.mkdir(exist_ok=False)
    reports={}
    for name in (['base'] if args.stage=='base' else ['h3','constant']):
        head=copy.deepcopy(base) if name=='base' else StopValueHead(base)
        if name=='constant':head.stop.weight.requires_grad_(False)
        params=[p for p in head.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=.0001 if name=='base' else .001)
        for _ in range(100):
            if name=='base':
                lc=-(target*head(cx,cm)[0].log_softmax(-1)).sum(-1);lr=-(rt*head(rx,rm)[0].log_softmax(-1)).sum(-1)
            else:
                lc=F.binary_cross_entropy_with_logits(stop_margin(head,cx,cm),target,reduction='none');lr=F.binary_cross_entropy_with_logits(stop_margin(head,rx,rm),rt,reduction='none')
            loss=.5*(cw*lc).sum()+.5*(rw*lr).sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        if name!='base':
            assert all(torch.equal(v,head.base.state_dict()[k]) for k,v in base.state_dict().items())
            with torch.no_grad():assert torch.equal(head(cx,cm)[0][:,:3],base(cx,cm)[0][:,:3])
        else:
            assert torch.equal(head.mean,base.mean) and torch.equal(head.scale,base.scale)
        path=out/f'{name}.pt';torch.save(head.state_dict(),path);reports[name]=dict(checkpoint=str(path),sha256=sha(path),parameters=sum(p.numel() for p in params),loss=float(loss.detach()))
    save(out/'manifest.json',dict(models=reports,source_tensor_sha256=sha(run/'states.pt'),reference_checkpoint_sha256=sha(spec['checkpoint']),corrections=len(labels),retention=len(retain),targets=evidence,plan=plan))
    save(out/'status.json',dict(phase='complete',pid=None));print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
