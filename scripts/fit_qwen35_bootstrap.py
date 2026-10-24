"""Fresh9B multi-position MLP initialization; never load a4B head."""
import hashlib,json
from pathlib import Path
from collections import defaultdict
import torch
from torch.nn import functional as F
from lsdrag.decision_state_learning import StateHead
from lsdrag.decision_policy_followup import state_weights
ROOT=Path('/content/local_fast/lsdrag-text-v1');REPO=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plan=json.loads((REPO/'lab/registry/qwen35_9b_h3_20261024/training_plan.json').read_text())
    for f,h in plan['bootstrap_source_sha256'].items():assert sha(REPO/f)==h
    run=ROOT/'results/qwen35_9b_teacher192_20261024';assert json.loads((run/'status.json').read_text())['phase']=='complete'
    spec=json.loads((run/'spec.json').read_text());records=torch.load(run/'teacher_states.pt',weights_only=True,map_location='cpu')
    assert {r['qid'] for r in records}==set(spec['qids'])==set(plan['train'])
    coverage=defaultdict(set)
    for r in records:coverage[int(r['action'])].add(r['qid']);assert r['pack'].shape==(37,4096)
    assert all(len(coverage[a])>=5 for a in range(4)),{a:len(v) for a,v in coverage.items()}
    torch.set_num_threads(2);torch.manual_seed(20261051)
    x=torch.stack([r['pack'] for r in records]).float();legal=torch.stack([r['legal'] for r in records]);act=torch.tensor([r['action'] for r in records]);w=state_weights([r['qid'] for r in records],'cpu')
    head=StateHead(dim=4096,inputs='multi',body='mlp');head.normalize_from_training(x)
    optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4)
    for _ in range(100):
        lp=head(x,legal)[0].log_softmax(-1);nll=-lp.gather(1,act[:,None]).squeeze(1);smooth=-lp.masked_fill(~legal,0).sum(1)/legal.sum(1)
        loss=(w*(.95*nll+.05*smooth)).sum();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
    out=ROOT/'results/qwen35_9b_bootstrap_20261024';out.mkdir(exist_ok=False);torch.save(head.state_dict(),out/'head.pt')
    with torch.no_grad():accuracy=float((head(x,legal)[0].argmax(-1)==act).float().mean())
    report=dict(phase='complete',pid=None,checkpoint=str(out/'head.pt'),sha256=sha(out/'head.pt'),teacher_sha256=sha(run/'teacher_states.pt'),states=len(records),train_questions=len(spec['qids']),coverage={a:len(v) for a,v in coverage.items()},training_accuracy=accuracy,method='Fresh4096hidden multi-position128MLP;100AdamW3e-4,seed20261051,.05label smoothing; train-only initialization, not outcome-trained H3')
    (out/'status.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
