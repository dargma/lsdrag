"""Training-only output-layer backtracking around the current policy."""
import torch
from lsdrag.decision_policy_followup import state_weights

@torch.no_grad()
def conservative_output_update(head, reference, pack, legal, qids, retention_mask,
                               kl_limit=.01, change_limit=.02, max_backtracks=10):
    assert kl_limit>=0 and 0<=change_limit<=1 and bool(retention_mask.any())
    weights=state_weights(qids,pack.device)
    before=reference(pack,legal)[0];logp=before.log_softmax(-1);p=logp.exp();actions=before.argmax(-1)
    origin={k:v.detach().clone() for k,v in reference.base.net[2].state_dict().items()}
    proposal={k:v.detach().clone() for k,v in head.base.net[2].state_dict().items()}
    trace=[]
    for power in range(max_backtracks+1):
        alpha=2.**(-power)
        head.base.net[2].load_state_dict({k:origin[k]+alpha*(proposal[k]-origin[k]) for k in origin})
        after=head(pack,legal)[0]
        kl=float((weights*(p*(logp-after.log_softmax(-1))).sum(-1)).sum())
        changed=float((after.argmax(-1)[retention_mask]!=actions[retention_mask]).float().mean())
        entry=dict(alpha=alpha,kl=kl,unsupported_change_fraction=changed)
        trace.append(entry)
        if kl<=kl_limit and changed<=change_limit:
            return dict(accepted=True,**entry,trace=trace)
    head.base.net[2].load_state_dict(origin)
    return dict(accepted=False,alpha=0.,kl=0.,unsupported_change_fraction=0.,trace=trace)
