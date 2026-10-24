import sys,hashlib
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_decision_quality_head_grpo import continuation_heads
from lsdrag.decision_value_head import DecisionValueHead
from lsdrag.decision_policy import distribution

def test_quality_head_continuation_updates_actor_preserves_aux_norm_reference(tmp_path):
    torch.manual_seed(7);torch.set_num_threads(2)
    initial=DecisionValueHead(body='mlp');x=torch.randn(4,5,2560);initial.normalize_from_training(x)
    path=tmp_path/'head.pt';torch.save(initial.state_dict(),path)
    spec=dict(kind='decision_value',body='mlp',checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    reference,head=continuation_heads(spec)
    assert all(torch.equal(v,head.state_dict()[k]) for k,v in initial.state_dict().items())
    assert {n for n,p in head.named_parameters() if p.requires_grad}=={'encoder.weight','encoder.bias','policy_weight','policy_bias'}
    legal=torch.ones(4,4,dtype=torch.bool);legal[:,2]=False
    action=torch.zeros(4,dtype=torch.long)
    old=distribution(reference(x,legal)[0],legal).log_prob(action).detach()
    before=distribution(head(x,legal)[0],legal).probs.detach()
    opt=torch.optim.AdamW(head.parameters(),lr=1e-4)
    ratio=(distribution(head(x,legal)[0],legal).log_prob(action)-old).exp()
    (-torch.minimum(ratio,ratio.clamp(.8,1.2))).mean().backward();opt.step()
    after=distribution(head(x,legal)[0],legal).probs.detach()
    assert after[:,0].mean()>before[:,0].mean()
    for k,v in initial.state_dict().items():
        assert torch.equal(reference.state_dict()[k],v)
        if not k.startswith(('encoder.','policy_')):assert torch.equal(head.state_dict()[k],v)
